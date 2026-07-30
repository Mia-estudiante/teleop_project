import argparse

import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, PoseArray

# pinocchio
import pinocchio as pin
from pinocchio.utils import *

import numpy as np

from teleop_manager.src.robot.h1_2_wrapper import H12Wrapper
from teleop_manager.src.robot.igris_c_wrapper import IgrisCWrapper
from teleop_manager.src.dex_retargeting.DexRetargeting import DexRetargeting
# from simulation.simulation_hand_node import MujocoSimulationHandNode

THRES = 3e-4

class TeleopUpperNode(Node):
    def __init__(self, robot_name):
        super().__init__('teleop_upper_node')
        self.get_logger().info('Teleop Upper Node has been started.')
        
        self.robot_name = robot_name
        self.head_goal = pin.SE3()
        self.l_goal = pin.SE3()
        self.r_goal = pin.SE3()
        self.l_qdes = np.zeros(7)
        self.r_qdes = np.zeros(7)
        self.lfingers_goal = []
        self.rfingers_goal = []
        self.lfingers_qdes = np.zeros(6)
        self.rfingers_qdes = np.zeros(6)

        if robot_name == 'h1_2':
            self.robot = H12Wrapper()
        elif robot_name == 'igris_c':
            self.robot = IgrisCWrapper()
            self.rviz_publisher = self.create_publisher(JointState, '/joint_states', 10)
            self.all_joint_names = self.robot.all_joint_names
            self.all_joint_names.remove("universe")
            self.hand_joint_names = [joint_name for joint_name in self.all_joint_names if "Left_" in joint_name or "Right_" in joint_name]

        self.dexretargeting = DexRetargeting(robot_name)
        # print("self.left_retargeting_index", self.dexretargeting.left_retargeting_index)
        # print("self.right_retargeting_index", self.dexretargeting.right_retargeting_index)

        self.publisher = self.create_publisher(Float64MultiArray, '/mujoco/controller', 10)
        self.subscriber = self.create_subscription(JointState, '/mujoco/joint_states', self.joint_state_callback, 10)
        self.head_subscriber = self.create_subscription(Pose, '/head', self.head_callback, 10)
        self.lwrist_subscriber = self.create_subscription(Pose, '/lwrist', self.lwrist_callback, 10)
        self.rwrist_subscriber = self.create_subscription(Pose, '/rwrist', self.rwrist_callback, 10)
        self.lfingers_subscriber = self.create_subscription(PoseArray, '/lfingers', self.lfingers_callback, 10)
        self.rfingers_subscriber = self.create_subscription(PoseArray, '/rfingers', self.rfingers_callback, 10)

        self.timer = self.create_timer(0.01, self.timer_callback)
        
        self.ctrlFlag = False           # joint_state 콜백이 처음으로 들어왔는지 여부
        ############
        self.head_ctrlFlag = False    
        self.lwrist_ctrlFlag = False    
        self.rwrist_ctrlFlag = False     
        self.lfingers_ctrlFlag = False     
        self.rfingers_ctrlFlag = False
        ############
        if robot_name == 'h1_2':
            self.lwrist_rot_diff = np.array([[1,0,0],[0,0,-1],[0,1,0]])
            self.rwrist_rot_diff = np.array([[1,0,0],[0,0,1],[0,-1,0]])
            self.lwrist_rot_diff_se3 = pin.SE3(self.lwrist_rot_diff, np.zeros(3))
            self.rwrist_rot_diff_se3 = pin.SE3(self.rwrist_rot_diff, np.zeros(3))
        elif robot_name == 'igris_c':
            self.lwrist_rot_diff = np.array([[1,0,0],[0,0,1],[0,-1,0]]).T @ np.array([[0,0,-1],[0,1,0],[1,0,0]])
            self.rwrist_rot_diff = np.array([[1,0,0],[0,0,-1],[0,1,0]]).T @ np.array([[0,0,-1],[0,1,0],[1,0,0]])
            self.lwrist_rot_diff_se3 = pin.SE3(self.lwrist_rot_diff, np.zeros(3))
            self.rwrist_rot_diff_se3 = pin.SE3(self.rwrist_rot_diff, np.zeros(3))

    def timer_callback(self):
        # qdes = np.zeros(26)
        if self.ctrlFlag:
            self.get_logger().info(f"Joint State Received.")
            if self.head_ctrlFlag and self.lwrist_ctrlFlag and self.rwrist_ctrlFlag and self.lfingers_ctrlFlag and self.rfingers_ctrlFlag:
                # ROBOT
                headMlwrist = self.robot.state.head_oMi.inverse() * self.robot.state.l_oMi
                headMrwrist = self.robot.state.head_oMi.inverse() * self.robot.state.r_oMi
                
                # USER
                headMl_target =  self.head_goal.inverse() * self.l_goal.copy() * self.lwrist_rot_diff_se3
                headMr_target =  self.head_goal.inverse() * self.r_goal.copy() * self.rwrist_rot_diff_se3
                
                # 1. 포즈 오차 계산 (목표와 현재의 차이)
                l_dMi = headMlwrist.inverse() * headMl_target
                r_dMi = headMrwrist.inverse() * headMr_target

                x_err_l = pin.log(l_dMi).vector
                x_err_r = pin.log(r_dMi).vector

                # 2. 기본 공식 (Primary Task만 적용)
                # qdot = J_inv * (Gain * error)
                l_qdot = np.linalg.pinv(self.robot.state.l_J) @ (1.5 * x_err_l)
                r_qdot = np.linalg.pinv(self.robot.state.r_J) @ (1.5 * x_err_r)

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

        arm_qdes = np.concatenate([
            self.l_qdes,
            self.r_qdes
        ])
        hand_qdes = np.concatenate([
            self.lfingers_qdes,
            self.rfingers_qdes
        ])
        if self.robot_name == "h1_2":
            self.publish_h1_2(
                arm_qdes,
                hand_qdes
            )
        elif self.robot_name == "igris_c":
            self.publish_igris_c(
                arm_qdes,
                hand_qdes
            )

    def publish_h1_2(self, arm_qdes, hand_qdes):
        qdes = np.zeros(26)
        qdes[:7] = arm_qdes[:7]
        qdes[7:13] = hand_qdes[:6]
        qdes[13:20] = arm_qdes[7:]
        qdes[20:] = hand_qdes[6:]
        msg = Float64MultiArray()
        msg.data = qdes.tolist()
        self.publisher.publish(msg)

    def publish_igris_c(self, arm_qdes, hand_qdes):
        # mujoco
        mujoco_msg = Float64MultiArray()
        mujoco_msg.data = arm_qdes.tolist()
        self.publisher.publish(mujoco_msg)

        # rviz
        self.rviz_publish(
            arm_qdes,
            hand_qdes
        )

    '''
    Mujoco JointState: 현재 robot state 정보를 받아서 H12Wrapper의 state에 반영
    '''
    # def upper_publish(self, qdes):
    #     upper_msg = Float64MultiArray()
    #     upper_msg.data = qdes.tolist()
    #     self.publisher.publish(upper_msg)
    
    def rviz_publish(self, arm_qdes, hand_qdes):
        rviz_msg = JointState()
        rviz_msg.header.stamp = self.get_clock().now().to_msg()
        rviz_msg.name = self.all_joint_names
        hand_msg = JointState()
        hand_msg.header.stamp = self.get_clock().now().to_msg()
        hand_msg.name = self.hand_joint_names
        
        q_out = [0.0] * len(self.all_joint_names)
        q_out[15:22] = arm_qdes[:7] # left wrist
        q_out[33:40] = arm_qdes[7:] # right wrist

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
        self.lwrist_ctrlFlag = True

    def rwrist_callback(self, msg: Pose):
        pos = np.array([msg.position.x, msg.position.y, msg.position.z, msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        self.r_goal = pin.XYZQUATToSE3(pos)
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robot", choices=['h1_2','igris_c'], default='h1_2'
    )
    args, ros_args = parser.parse_known_args()
    rclpy.init(args=ros_args)
    node = TeleopUpperNode(args.robot)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
