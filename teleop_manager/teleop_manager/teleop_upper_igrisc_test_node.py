import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import Float64MultiArray, Bool
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, PoseArray

# pinocchio
import pinocchio as pin
from pinocchio.utils import *

import numpy as np

from teleop_manager.src.robot.igris_c_wrapper import IgrisCWrapper
from teleop_manager.src.dex_retargeting.DexRetargeting import DexRetargeting

THRES = 3e-4

class TeleopUpperIgrisCTestNode(Node):
    def __init__(self):
        super().__init__('teleop_upper_igris_c_node')
        self.get_logger().info('Teleop Upper IgrisC Node has been started.')

        self.head_goal = pin.SE3()
        self.l_goal = pin.SE3()
        self.r_goal = pin.SE3()
        self.l_qdes = np.zeros(7)
        self.r_qdes = np.zeros(7)
        self.qdes_initialized = False

        # Calibration offsets — captured at first teleop entry to avoid jumps
        self.head_goal_origin = None
        self.l_goal_origin = None
        self.r_goal_origin = None

        self.lfingers_goal = []
        self.rfingers_goal = []
        self.lfingers_qdes = np.zeros(6)
        self.rfingers_qdes = np.zeros(6)

        self.init_qdes = np.zeros(14)
        self.init_qdes[3] = -np.pi/2
        self.init_qdes[10] = -np.pi/2

        self.robot = IgrisCWrapper()
        self.all_joint_names = self.robot.all_joint_names
        self.all_joint_names.remove("universe")
        print(self.all_joint_names)
        self.hand_joint_names = [j for j in self.all_joint_names if "Left_" in j or "Right_" in j]
        self.dexretargeting = DexRetargeting('igris_c')

        # === Publishers / Subscribers ===
        # Best-effort + depth=1 keeps only the latest command (avoid stale build-up)
        ctrl_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,  # ← BEST_EFFORT에서 변경
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.publisher      = self.create_publisher(Float64MultiArray, '/mujoco/controller', ctrl_qos)
        self.hand_publisher = self.create_publisher(Float64MultiArray, '/mujoco/hand_controller', ctrl_qos)
        self.rviz_publisher = self.create_publisher(JointState, '/joint_states', 10)
        self.flag_publisher = self.create_publisher(Bool, '/mujoco/flag', 10)

        self.subscriber          = self.create_subscription(JointState, '/mujoco/joint_states', self.joint_state_callback, 10)
        self.head_subscriber     = self.create_subscription(Pose,      '/head',     self.head_callback,     10)
        self.lwrist_subscriber   = self.create_subscription(Pose,      '/lwrist',   self.lwrist_callback,   10)
        self.rwrist_subscriber   = self.create_subscription(Pose,      '/rwrist',   self.rwrist_callback,   10)
        self.lfingers_subscriber = self.create_subscription(PoseArray, '/lfingers', self.lfingers_callback, 10)
        self.rfingers_subscriber = self.create_subscription(PoseArray, '/rfingers', self.rfingers_callback, 10)

        # 100Hz timer; actual dt is measured (handles jitter)
        self.timer = self.create_timer(0.01, self.timer_callback)
        self.last_callback_time = None

        # === Flags ===
        self.ctrlFlag         = False
        self.initCtrlFlag     = True
        self.initCtrlFlag2    = True
        self.lwrist_ctrlFlag  = False
        self.rwrist_ctrlFlag  = False
        self.lfingers_ctrlFlag = False
        self.rfingers_ctrlFlag = False
        self.head_ctrlFlag    = False

        # === Frame correction matrices ===
        self.lwrist_rot_diff = np.array([[1,0,0],[0,0,1],[0,-1,0]]).T @ np.array([[0,0,-1],[0,1,0],[1,0,0]])
        self.rwrist_rot_diff = np.array([[1,0,0],[0,0,-1],[0,1,0]]).T @ np.array([[0,0,-1],[0,1,0],[1,0,0]])
        self.lwrist_rot_diff_se3 = pin.SE3(self.lwrist_rot_diff, np.zeros(3))
        self.rwrist_rot_diff_se3 = pin.SE3(self.rwrist_rot_diff, np.zeros(3))

    def timer_callback(self):
        # === Measure real dt (handles timer jitter) ===
        now = self.get_clock().now()
        if self.last_callback_time is None:
            dt = 0.01
        else:
            dt = (now - self.last_callback_time).nanoseconds / 1e9
            dt = max(0.001, min(dt, 0.05))   # clamp to safe range
        self.last_callback_time = now

        qdes = self.robot.state.q[:14].copy()
        hand_qdes = np.zeros(12)

        if self.ctrlFlag:
            self.get_logger().info(f"Joint State Received.")

            if (self.head_ctrlFlag and self.lwrist_ctrlFlag and self.rwrist_ctrlFlag
                    and self.lfingers_ctrlFlag and self.rfingers_ctrlFlag):

                # === First entry: capture origin pose to avoid jumps ===
                if not self.qdes_initialized:
                    print("init qdes, current robot q:", self.robot.state.q)
                    self.l_qdes = qdes[:7].copy()
                    self.r_qdes = qdes[7:].copy()

                    # Snapshot user's pose as the new origin
                    self.head_goal_origin = self.head_goal.copy()
                    self.l_goal_origin    = self.l_goal.copy()
                    self.r_goal_origin    = self.r_goal.copy()

                    self.qdes_initialized = True

                # === ROBOT current poses (head frame) ===
                headMlwrist = self.robot.state.head_oMi.inverse() * self.robot.state.l_oMi
                headMrwrist = self.robot.state.head_oMi.inverse() * self.robot.state.r_oMi

                # === USER target poses (relative to origin pose, head frame) ===
                # User's hand displacement from origin, expressed in original head frame
                headMl_target = self.head_goal.inverse() * self.l_goal * self.lwrist_rot_diff_se3
                headMr_target = self.head_goal.inverse() * self.r_goal * self.rwrist_rot_diff_se3

                headMl_target.translation = headMl_target.translation.copy() * 0.8
                headMr_target.translation = headMr_target.translation.copy() * 0.8

                # === Pose error → joint velocity (DLS IK) ===
                l_dMi = headMlwrist.inverse() * headMl_target
                r_dMi = headMrwrist.inverse() * headMr_target

                x_err_l = pin.log(l_dMi).vector
                x_err_r = pin.log(r_dMi).vector

                lambda_val = 0.05

                l_J = self.robot.state.l_J
                l_J_dls = l_J.T @ np.linalg.inv(l_J @ l_J.T + (lambda_val**2) * np.eye(l_J.shape[0]))
                l_qdot = l_J_dls @ (1.5 * x_err_l)

                r_J = self.robot.state.r_J
                r_J_dls = r_J.T @ np.linalg.inv(r_J @ r_J.T + (lambda_val**2) * np.eye(r_J.shape[0]))
                r_qdot = r_J_dls @ (1.5 * x_err_r)

                # === Velocity clamp (prevents jumps from sudden VR pose changes) ===
                MAX_QDOT = 3.0   # rad/s per joint
                l_qdot = np.clip(l_qdot, -MAX_QDOT, MAX_QDOT)
                r_qdot = np.clip(r_qdot, -MAX_QDOT, MAX_QDOT)

                # === Integrate using REAL dt (not hardcoded 0.01) ===
                self.l_qdes += l_qdot * dt
                self.r_qdes += r_qdot * dt

                # === Hand retargeting ===
                l_wristMl_fingers = []
                r_wristMl_fingers = []
                for lf in self.lfingers_goal:
                    l_wristMl_fingers.append((self.l_goal.inverse() * lf).translation.tolist())
                for rf in self.rfingers_goal:
                    r_wristMl_fingers.append((self.r_goal.inverse() * rf).translation.tolist())

                self.lfingers_qdes = self.dexretargeting.retarget_ref(
                    self.dexretargeting.left_retargeting,
                    np.array(l_wristMl_fingers)
                )[self.dexretargeting.left_retargeting_index]

                self.rfingers_qdes = self.dexretargeting.retarget_ref(
                    self.dexretargeting.right_retargeting,
                    np.array(r_wristMl_fingers)
                )[self.dexretargeting.right_retargeting_index]

                qdes = np.append(self.l_qdes, self.r_qdes)
                hand_qdes = np.append(self.lfingers_qdes, self.rfingers_qdes)

        print("current qdes:", qdes)
        print("current robot q:", self.robot.state.q)

        self.arm_publish(qdes)
        # self.hand_publish(hand_qdes)
        # self.rviz_publish(qdes, hand_qdes)

    def arm_publish(self, qdes):
        msg = Float64MultiArray()
        msg.data = qdes.tolist()
        self.publisher.publish(msg)

    def hand_publish(self, qdes):
        msg = Float64MultiArray()
        msg.data = qdes.tolist()
        self.hand_publisher.publish(msg)

    def rviz_publish(self, qdes, hand_qdes):
        rviz_msg = JointState()
        rviz_msg.header.stamp = self.get_clock().now().to_msg()
        rviz_msg.name = self.all_joint_names

        hand_msg = JointState()
        hand_msg.header.stamp = self.get_clock().now().to_msg()
        hand_msg.name = self.hand_joint_names

        q_out = [0.0] * len(self.all_joint_names)
        q_out[15:22] = qdes[:7]
        q_out[33:40] = qdes[7:]

        q_hand_out = [0.0] * len(self.hand_joint_names)

        def map_to_all(side, qpos, joint_names):
            for i, name in enumerate(joint_names):
                full = side + name
                if full in self.all_joint_names:
                    q_out[self.all_joint_names.index(full)] = float(qpos[i])
                    q_hand_out[self.hand_joint_names.index(full)] = float(qpos[i])

        map_to_all('Left_',  hand_qdes[:6], self.dexretargeting.left_retargeting.optimizer.target_joint_names)
        map_to_all('Right_', hand_qdes[6:], self.dexretargeting.right_retargeting.optimizer.target_joint_names)

        mimic_map = {
            'Left_2_Joint_Thumb_Distal':   'Left_1_Joint_Thumb_Middle',
            'Left_4_Joint_Index_Distal':   'Left_3_Joint_Index_Middle',
            'Left_6_Joint_Middle_Distal':  'Left_5_Joint_Middle_Middle',
            'Left_8_Joint_Ring_Distal':    'Left_7_Joint_Ring_Middle',
            'Left_10_Joint_Little_Distal': 'Left_9_Joint_Little_Middle',
            'Right_2_Joint_Thumb_Distal':   'Right_1_Joint_Thumb_Middle',
            'Right_4_Joint_Index_Distal':   'Right_3_Joint_Index_Middle',
            'Right_6_Joint_Middle_Distal':  'Right_5_Joint_Middle_Middle',
            'Right_8_Joint_Ring_Distal':    'Right_7_Joint_Ring_Middle',
            'Right_10_Joint_Little_Distal': 'Right_9_Joint_Little_Middle',
        }
        for child, parent in mimic_map.items():
            p_idx = self.all_joint_names.index(parent)
            p_hand_idx = self.hand_joint_names.index(parent)
            q_out[self.all_joint_names.index(child)] = q_out[p_idx]
            q_hand_out[self.hand_joint_names.index(child)] = q_hand_out[p_hand_idx]

        hand_msg.position = q_hand_out
        rviz_msg.position = q_out
        self.rviz_publisher.publish(rviz_msg)

    def flag_publish(self, flag):
        msg = Bool()
        msg.data = not flag
        self.flag_publisher.publish(msg)

    # === Subscribers ===
    def joint_state_callback(self, msg: JointState):
        self.robot.state.q = np.array(msg.position)
        self.robot.state.v = np.array(msg.velocity)
        self.robot.computeAllTerms()
        self.ctrlFlag = True

    def head_callback(self, msg: Pose):
        pos = np.array([msg.position.x, msg.position.y, msg.position.z,
                        msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        self.head_goal = pin.XYZQUATToSE3(pos)
        self.head_ctrlFlag = True

    def lwrist_callback(self, msg: Pose):
        pos = np.array([msg.position.x, msg.position.y, msg.position.z,
                        msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        self.l_goal = pin.XYZQUATToSE3(pos)
        self.lwrist_ctrlFlag = True

    def rwrist_callback(self, msg: Pose):
        pos = np.array([msg.position.x, msg.position.y, msg.position.z,
                        msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        self.r_goal = pin.XYZQUATToSE3(pos)
        self.rwrist_ctrlFlag = True

    def lfingers_callback(self, msg: PoseArray):
        out = []
        for p in msg.poses:
            arr = np.array([p.position.x, p.position.y, p.position.z,
                            p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w])
            out.append(pin.XYZQUATToSE3(arr))
        self.lfingers_goal = out
        self.lfingers_ctrlFlag = True

    def rfingers_callback(self, msg: PoseArray):
        out = []
        for p in msg.poses:
            arr = np.array([p.position.x, p.position.y, p.position.z,
                            p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w])
            out.append(pin.XYZQUATToSE3(arr))
        self.rfingers_goal = out
        self.rfingers_ctrlFlag = True


def main():
    rclpy.init()
    node = TeleopUpperIgrisCTestNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()