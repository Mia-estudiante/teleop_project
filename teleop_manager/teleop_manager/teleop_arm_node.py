import os

import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose

# pinocchio
import pinocchio as pin
from pinocchio.utils import *

import numpy as np

from teleop_manager.src.robot.h1_2_wrapper import H12Wrapper

THRES = 3e-4

class TeleopArmNode(Node):
    def __init__(self):
        super().__init__('teleop_arm_node')
        self.get_logger().info('Teleop Arm Node has been started.')

        self.l_goal = pin.SE3()
        self.r_goal = pin.SE3()
        # self.head_goal = pin.SE3()
        self.l_qdes = np.zeros(7)
        self.r_qdes = np.zeros(7)
        self.init_qdes = np.zeros(14)
        # self.head_qdes = np.zeros(7)

        self.robot = H12Wrapper()
        self.robot2 = H12Wrapper()

        self.publisher = self.create_publisher(Float64MultiArray, '/mujoco/controller', 10)
        self.subscriber = self.create_subscription(JointState, '/mujoco/joint_states', self.joint_state_callback, 10)
        self.lwrist_subscriber = self.create_subscription(Pose, '/lwrist', self.lwrist_callback, 10)
        self.rwrist_subscriber = self.create_subscription(Pose, '/rwrist', self.rwrist_callback, 10)
        # self.head_subscriber = self.create_subscription(Pose, '/head', self.head_callback, 10)

        self.timer = self.create_timer(0.01, self.timer_callback)
        
        self.ctrlFlag = False           # mujoco의 joinstate
        self.initCtrlFlag = True        # teleop 시작 전 초기화 플래그
        self.initCtrlFlag2 = True        # teleop 시작 전 초기화 플래그
        self.lwrist_ctrlFlag = False    
        self.rwrist_ctrlFlag = False     
        # self.head_ctrlFlag = False

    def timer_callback(self):
        qdes = np.zeros(14)
        if self.ctrlFlag:
            if self.initCtrlFlag:
                self.get_logger().info(f"Initializing...")
                if np.linalg.norm(qdes - self.init_qdes) < THRES:
                    qdes = self.init_qdes.copy()
                    self.initCtrlFlag = False
                    self.get_logger().info(f"Initializing Done. Moving to teleoperation...")
            else:
                if self.lwrist_ctrlFlag and self.rwrist_ctrlFlag:
                    if self.initCtrlFlag2:
                        self.l_goal = self.robot.state.l_oMi.copy()
                        self.r_goal = self.robot.state.r_oMi.copy() 
                        self.l_qdes = self.robot.state.q[:7].copy()
                        self.r_qdes = self.robot.state.q[7:].copy()
                        self.initCtrlFlag2 = False
                    else: 
                        # 1. 포즈 오차 계산 (목표와 현재의 차이)
                        l_dMi = self.robot.state.l_oMi.inverse() * self.l_goal
                        r_dMi = self.robot.state.r_oMi.inverse() * self.r_goal

                        x_err_l = pin.log(l_dMi)
                        x_err_r = pin.log(r_dMi)
                        # print("x_err_l:", x_err_l)
                        # 2. 기본 공식 (Primary Task만 적용)
                        # qdot = J_inv * (Gain * error)
                        qdot_l = np.linalg.pinv(self.robot.state.l_J) @ (10 * x_err_l)
                        qdot_r = np.linalg.pinv(self.robot.state.r_J) @ (10 * x_err_r)

                        # 3. 적분하여 명령 생성
                        # print("l_qdes:", self.l_qdes)
                        # print("qdot_l:", qdot_l)
                        self.l_qdes += qdot_l * 0.01
                        self.r_qdes += qdot_r * 0.01

                qdes = np.append(self.l_qdes, self.r_qdes)
                print("Desired joint positions:", qdes)
        self.arm_publish(qdes)

    '''
    Mujoco JointState: 현재 robot state 정보를 받아서 H12Wrapper의 state에 반영
    '''
    def arm_publish(self, qdes=None):
        arm_msg = Float64MultiArray()
        arm_msg.data = qdes.tolist()
        # arm_msg.data = self.robot.state.q.tolist()
        # print("Publishing desired joint positions:", arm_msg.data)
        self.publisher.publish(arm_msg)

    def joint_state_callback(self, msg: JointState):
        self.robot.state.q = np.array(msg.position) # 14
        self.robot.state.v = np.array(msg.velocity) # 14
        self.robot.computeAllTerms()
        self.get_logger().warn(f"Current joint positions: {self.robot.state.q}")
        self.ctrlFlag = True

    def lwrist_callback(self, msg: Pose):
        pos = np.array([msg.position.x, msg.position.y, msg.position.z, msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        self.l_goal = pin.XYZQUATToSE3(pos)
        self.get_logger().warn(f"Target Left : {self.l_goal}")
        self.lwrist_ctrlFlag = True

    def rwrist_callback(self, msg: Pose):
        pos = np.array([msg.position.x, msg.position.y, msg.position.z, msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        self.r_goal = pin.XYZQUATToSE3(pos)
        self.get_logger().warn(f"Target Right : {self.r_goal}")
        self.rwrist_ctrlFlag = True

    def head_callback(self, msg: Pose):
        pos = np.array([msg.position.x, msg.position.y, msg.position.z, msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        self.head_goal = pin.XYZQUATToSE3(pos)
        self.head_ctrlFlag = True

def main():
    rclpy.init()
    node = TeleopArmNode()
    rclpy.spin(node)
    # node.main()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
