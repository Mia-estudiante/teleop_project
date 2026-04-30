import rclpy
from rclpy.node import Node

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

class TeleopUpperIgrisCNode(Node):
    def __init__(self):
        super().__init__('teleop_upper_igris_c_node')
        self.get_logger().info('Teleop Upper IgrisC Node has been started.')

        self.head_goal = pin.SE3()
        self.l_goal = pin.SE3()
        self.r_goal = pin.SE3()
        self.l_qdes = np.zeros(7)
        self.r_qdes = np.zeros(7)
        self.qdes_initialized = False   # 추가

        self.lfingers_goal = []
        self.rfingers_goal = []
        self.lfingers_qdes = np.zeros(6)
        self.rfingers_qdes = np.zeros(6)

        self.init_qdes = np.zeros(14) 
        self.init_qdes[3] = -np.pi/2 # Joint_Elbow_Pitch_Left
        self.init_qdes[10] = -np.pi/2 # Joint_Elbow_Pitch_Right

        self.robot = IgrisCWrapper()
        self.all_joint_names = self.robot.all_joint_names
        self.all_joint_names.remove("universe")
        print(self.all_joint_names)
        self.hand_joint_names = [joint_name for joint_name in self.all_joint_names if "Left_" in joint_name or "Right_" in joint_name]
        self.dexretargeting = DexRetargeting('igris_c') # 추후, params 로 값을 받도록 할 것

        self.publisher = self.create_publisher(Float64MultiArray, '/mujoco/controller', 10)
        self.hand_publisher = self.create_publisher(Float64MultiArray, '/mujoco/hand_controller', 10)
        self.rviz_publisher = self.create_publisher(JointState, '/joint_states', 10)
        self.flag_publisher = self.create_publisher(Bool, '/mujoco/flag', 10)
        self.subscriber = self.create_subscription(JointState, '/mujoco/joint_states', self.joint_state_callback, 10)
        self.head_subscriber = self.create_subscription(Pose, '/head', self.head_callback, 10)
        self.lwrist_subscriber = self.create_subscription(Pose, '/lwrist', self.lwrist_callback, 10)
        self.rwrist_subscriber = self.create_subscription(Pose, '/rwrist', self.rwrist_callback, 10)
        self.lfingers_subscriber = self.create_subscription(PoseArray, '/lfingers', self.lfingers_callback, 10)
        self.rfingers_subscriber = self.create_subscription(PoseArray, '/rfingers', self.rfingers_callback, 10)

        self.timer = self.create_timer(0.01, self.timer_callback)
        
        self.ctrlFlag = False           
        self.initCtrlFlag = True        # teleop 시작 전 초기화 플래그
        self.initCtrlFlag2 = True       # teleop 시작 전 초기화 플래그
        self.lwrist_ctrlFlag = False    
        self.rwrist_ctrlFlag = False     
        self.lfingers_ctrlFlag = False
        self.rfingers_ctrlFlag = False
        self.head_ctrlFlag = False

        self.lwrist_rot_diff = np.array([[1,0,0],[0,0,1],[0,-1,0]]).T @ np.array([[0,0,-1],[0,1,0],[1,0,0]])
        self.rwrist_rot_diff = np.array([[1,0,0],[0,0,-1],[0,1,0]]).T @ np.array([[0,0,-1],[0,1,0],[1,0,0]])
        self.lwrist_rot_diff_se3 = pin.SE3(self.lwrist_rot_diff, np.zeros(3))
        self.rwrist_rot_diff_se3 = pin.SE3(self.rwrist_rot_diff, np.zeros(3))

        self.test_rad = -1.57

    def timer_callback(self):
        qdes = self.robot.state.q[:14].copy() # np.zeros(14)
        hand_qdes = np.zeros(12)

        if self.ctrlFlag:
            self.get_logger().info(f"Joint State Received.")
            # qdes = self.init_qdes.copy() # init 자세 만들어주기 위함
            if self.head_ctrlFlag and self.lwrist_ctrlFlag and self.rwrist_ctrlFlag and self.lfingers_ctrlFlag and self.rfingers_ctrlFlag:
                # 첫 진입 시 현재 로봇 q로 초기화 (점프 방지)
                if not self.qdes_initialized:
                    print("init qdes, current robot q:", self.robot.state.q)
                    self.l_qdes = qdes[:7].copy()
                    self.r_qdes = qdes[7:].copy()
                    # qdes = np.append(self.l_qdes, self.r_qdes)
                    self.qdes_initialized = True
                # if np.linalg.norm(self.robot.state.q - qdes) < THRES:
                #     return
                # ROBOT
                headMlwrist = self.robot.state.head_oMi.inverse() * self.robot.state.l_oMi
                headMrwrist = self.robot.state.head_oMi.inverse() * self.robot.state.r_oMi
                
                # USER
                headMl_target =  self.head_goal.inverse() * self.l_goal * self.lwrist_rot_diff_se3
                headMr_target =  self.head_goal.inverse() * self.r_goal * self.rwrist_rot_diff_se3
                
                headMl_target.translation = headMl_target.translation.copy() * 0.8 #+ np.array([0,0,0.05])
                headMr_target.translation = headMr_target.translation.copy() * 0.8 #+ np.array([0,0,0.05])

                # 1. 포즈 오차 계산 (목표와 현재의 차이)
                l_dMi = headMlwrist.inverse() * headMl_target
                r_dMi = headMrwrist.inverse() * headMr_target

                x_err_l = pin.log(l_dMi).vector
                x_err_r = pin.log(r_dMi).vector

                # 1. Damping Factor 설정 (보통 0.01 ~ 0.1 사이에서 튜닝)
                lambda_val = 0.05

                # 2. Left Hand DLS Inverse
                l_J = self.robot.state.l_J
                l_Identity = np.eye(l_J.shape[0])  # Task space 차원 (보통 6x6)
                # DLS 수식: J^T * (J * J^T + lambda^2 * I)^-1
                l_J_dls = l_J.T @ np.linalg.inv(l_J @ l_J.T + (lambda_val**2) * l_Identity)
                l_qdot = l_J_dls @ (1.5 * x_err_l)

                # 3. Right Hand DLS Inverse
                r_J = self.robot.state.r_J
                r_Identity = np.eye(r_J.shape[0])
                r_J_dls = r_J.T @ np.linalg.inv(r_J @ r_J.T + (lambda_val**2) * r_Identity)
                r_qdot = r_J_dls @ (1.5 * x_err_r)

                # 3. 적분하여 명령 생성
                self.l_qdes += l_qdot * 0.01
                self.r_qdes += r_qdot * 0.01

                l_wristMl_fingers = []
                r_wristMl_fingers = []
                
                for lfinger in self.lfingers_goal:
                    lwristMfingers = (self.l_goal).inverse() * lfinger
                    l_wristMl_fingers.append(lwristMfingers.translation.tolist())
                for rfinger in self.rfingers_goal:
                    rwristMfingers = (self.r_goal).inverse() * rfinger
                    r_wristMl_fingers.append(rwristMfingers.translation.tolist())

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
        
        # qdes = np.array([0, 0, 0, self.test_rad, 0, 0, 0,
        #                       0, 0, 0, self.test_rad, 0, 0, 0])
        print("current qdes:", qdes)
        print("current robot q:", self.robot.state.q)

        self.arm_publish(qdes)
        # self.test_rad += 0.01
        self.hand_publish(hand_qdes)
        self.rviz_publish(qdes, hand_qdes)

    def arm_publish(self, qdes):
        arm_msg = Float64MultiArray()
        arm_msg.data = qdes.tolist()
        self.publisher.publish(arm_msg)
        
    def hand_publish(self, qdes):
        hand_msg = Float64MultiArray()
        hand_msg.data = qdes.tolist()
        self.hand_publisher.publish(hand_msg)

    def rviz_publish(self, qdes, hand_qdes):
        rviz_msg = JointState()
        rviz_msg.header.stamp = self.get_clock().now().to_msg()
        rviz_msg.name = self.all_joint_names
        hand_msg = JointState()
        hand_msg.header.stamp = self.get_clock().now().to_msg()
        hand_msg.name = self.hand_joint_names
        
        q_out = [0.0] * len(self.all_joint_names)
        q_out[15:22] = qdes[:7] # left wrist
        q_out[33:40] = qdes[7:] # right wrist

        q_hand_out = [0.0] * len(self.hand_joint_names)

        def map_to_all(side, qpos, joint_names):
            for i, name in enumerate(joint_names):
                if side+name in self.all_joint_names:
                    q_out[self.all_joint_names.index(side+name)] = float(qpos[i])
                    q_hand_out[self.hand_joint_names.index(side+name)] = float(qpos[i])
                    # print("Mapping:", side+name, "->", self.all_joint_names.index(side+name))
        map_to_all('Left_', hand_qdes[:6], self.dexretargeting.left_retargeting.optimizer.target_joint_names)
        map_to_all('Right_', hand_qdes[6:], self.dexretargeting.right_retargeting.optimizer.target_joint_names)
    
        mimic_map = {
            'Left_2_Joint_Thumb_Distal': 'Left_1_Joint_Thumb_Middle',
            'Left_4_Joint_Index_Distal': 'Left_3_Joint_Index_Middle',
            'Left_6_Joint_Middle_Distal': 'Left_5_Joint_Middle_Middle',
            'Left_8_Joint_Ring_Distal': 'Left_7_Joint_Ring_Middle',
            'Left_10_Joint_Little_Distal': 'Left_9_Joint_Little_Middle',
            'Right_2_Joint_Thumb_Distal': 'Right_1_Joint_Thumb_Middle',
            'Right_4_Joint_Index_Distal': 'Right_3_Joint_Index_Middle',
            'Right_6_Joint_Middle_Distal': 'Right_5_Joint_Middle_Middle',
            'Right_8_Joint_Ring_Distal': 'Right_7_Joint_Ring_Middle',
            'Right_10_Joint_Little_Distal': 'Right_9_Joint_Little_Middle'
        }
        for child, parent in mimic_map.items():
            p_idx = self.all_joint_names.index(parent)
            p_hand_idx = self.hand_joint_names.index(parent)
            q_out[self.all_joint_names.index(child)] = q_out[p_idx]
            q_hand_out[self.hand_joint_names.index(child)] = q_hand_out[p_hand_idx]

        hand_msg.position = q_hand_out
        # self.hand_publisher.publish(hand_msg)
        rviz_msg.position = q_out
        self.rviz_publisher.publish(rviz_msg)

    def flag_publish(self, flag):
        flag_msg = Bool()
        flag_msg.data = not flag
        self.flag_publisher.publish(flag_msg)

    def joint_state_callback(self, msg: JointState):
        self.robot.state.q = np.array(msg.position) # 14
        self.robot.state.v = np.array(msg.velocity) # 14
        self.robot.computeAllTerms()
        self.ctrlFlag = True

    def head_callback(self, msg: Pose):
        pos = np.array([msg.position.x, msg.position.y, msg.position.z, msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        self.head_goal = pin.XYZQUATToSE3(pos)
        self.head_ctrlFlag = True

    def lwrist_callback(self, msg: Pose):
        pos = np.array([msg.position.x, msg.position.y, msg.position.z, msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        self.l_goal = pin.XYZQUATToSE3(pos)
        # self.l_goal.rotation = self.l_goal.rotation @ self.lwrist_rot_diff
        self.lwrist_ctrlFlag = True

    def rwrist_callback(self, msg: Pose):
        pos = np.array([msg.position.x, msg.position.y, msg.position.z, msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        self.r_goal = pin.XYZQUATToSE3(pos)
        # self.r_goal.rotation = self.r_goal.rotation @ self.rwrist_rot_diff
        self.rwrist_ctrlFlag = True
    
    def lfingers_callback(self, msg: PoseArray):
        lfingers_goal = []
        for i in range(len(msg.poses)):
            pos = np.array([msg.poses[i].position.x, msg.poses[i].position.y, msg.poses[i].position.z, msg.poses[i].orientation.x, msg.poses[i].orientation.y, msg.poses[i].orientation.z, msg.poses[i].orientation.w])
            lfingers_goal.append(pin.XYZQUATToSE3(pos))
        self.lfingers_goal = lfingers_goal
        self.lfingers_ctrlFlag = True

    def rfingers_callback(self, msg: PoseArray):
        rfingers_goal = []
        for i in range(len(msg.poses)):
            pos = np.array([msg.poses[i].position.x, msg.poses[i].position.y, msg.poses[i].position.z, msg.poses[i].orientation.x, msg.poses[i].orientation.y, msg.poses[i].orientation.z, msg.poses[i].orientation.w])
            rfingers_goal.append(pin.XYZQUATToSE3(pos))
        self.rfingers_goal = rfingers_goal
        self.rfingers_ctrlFlag = True   

def main():
    rclpy.init()
    node = TeleopUpperIgrisCNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
