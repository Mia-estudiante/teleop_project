import os

import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64MultiArray, Bool
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose

# pinocchio
import pinocchio as pin
from pinocchio.utils import *

import numpy as np

from teleop_manager.src.robot.h1_2_wrapper import H12Wrapper

THRES = 3e-4

class TeleopArmDataNode(Node):
    def __init__(self):
        super().__init__('teleop_arm_data_node')
        self.get_logger().info('Teleop Arm Data Node has been started.')

        self.head_goal = pin.SE3()
        self.l_goal = pin.SE3()
        self.r_goal = pin.SE3()
        self.l_qdes = np.zeros(7)
        self.r_qdes = np.zeros(7)
        self.init_qdes = np.zeros(14)

        self.robot = H12Wrapper()

        self.publisher = self.create_publisher(Float64MultiArray, '/mujoco/controller', 10)
        self.flag_publisher = self.create_publisher(Bool, '/mujoco/flag', 10)
        self.subscriber = self.create_subscription(JointState, '/mujoco/joint_states', self.joint_state_callback, 10)
        self.head_subscriber = self.create_subscription(Pose, '/head', self.head_callback, 10)
        self.lwrist_subscriber = self.create_subscription(Pose, '/lwrist', self.lwrist_callback, 10)
        self.rwrist_subscriber = self.create_subscription(Pose, '/rwrist', self.rwrist_callback, 10)

        self.timer = self.create_timer(0.01, self.timer_callback)
        
        self.ctrlFlag = False           
        self.initCtrlFlag = True        # teleop 시작 전 초기화 플래그
        self.initCtrlFlag2 = True       # teleop 시작 전 초기화 플래그
        self.lwrist_ctrlFlag = False    
        self.rwrist_ctrlFlag = False     
        self.head_ctrlFlag = False

    def timer_callback(self):
        qdes = np.zeros(14)
        if self.ctrlFlag:
            if self.initCtrlFlag:
                self.get_logger().info(f"Initializing...")
                if np.linalg.norm(self.robot.state.q - self.init_qdes) < THRES:
                    qdes = self.init_qdes.copy()
                    self.initCtrlFlag = False
                    self.get_logger().info(f"Initializing Done. Moving to teleoperation...")
            else:
                if self.lwrist_ctrlFlag and self.rwrist_ctrlFlag and self.head_ctrlFlag:
                    if self.initCtrlFlag2:
                        self.l_qdes = self.robot.state.q[:7].copy()
                        self.r_qdes = self.robot.state.q[7:].copy()
                        self.initCtrlFlag2 = False
                    else:
                        # ROBOT
                        headMlwrist = self.robot.state.head_oMi.inverse() * self.robot.state.l_oMi
                        headMrwrist = self.robot.state.head_oMi.inverse() * self.robot.state.r_oMi
                        
                        # USER
                        headMl_target =  self.head_goal.inverse() * self.l_goal
                        headMr_target =  self.head_goal.inverse() * self.r_goal
                        
                        # 1. 포즈 오차 계산 (목표와 현재의 차이)
                        # l_dMi = self.robot.state.l_oMi.inverse() * self.l_goal
                        # r_dMi = self.robot.state.r_oMi.inverse() * self.r_goal
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

                qdes = np.append(self.l_qdes, self.r_qdes)
        # qdes[:7] = np.zeros(7)  # 왼손 고정
        # qdes = np.zeros((14))
        self.arm_publish(qdes)
        self.flag_publish(self.initCtrlFlag)

    '''
    Mujoco JointState: 현재 robot state 정보를 받아서 H12Wrapper의 state에 반영
    '''
    def arm_publish(self, qdes):
        arm_msg = Float64MultiArray()
        arm_msg.data = qdes.tolist()
        self.publisher.publish(arm_msg)
    
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
        self.lwrist_ctrlFlag = True

    def rwrist_callback(self, msg: Pose):
        pos = np.array([msg.position.x, msg.position.y, msg.position.z, msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        self.r_goal = pin.XYZQUATToSE3(pos)
        self.rwrist_ctrlFlag = True

def main():
    rclpy.init()
    node = TeleopArmDataNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
