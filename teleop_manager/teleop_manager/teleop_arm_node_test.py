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

class TeleopArmNodeTest(Node):
    def __init__(self):
        super().__init__('teleop_arm_node')
        self.get_logger().info('Teleop Arm Node Test has been started.')

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
        self.step = 0

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
                if self.lwrist_ctrlFlag and self.rwrist_ctrlFlag:
                    if self.initCtrlFlag2:
                        self.l_goal = self.robot.state.l_oMi.copy()
                        self.r_goal = self.robot.state.r_oMi.copy() 
                        self.l_qdes = self.robot.state.q[:7].copy()
                        self.r_qdes = self.robot.state.q[7:].copy()

                        self.initCtrlFlag2 = False
                    else:
                        # 1. 포즈 오차 계산 (Local Frame 기준)
                        # 공식: log(현재_역행렬 * 목표)
                        # 현재 손목(oMi) 좌표계에서 목표(l_goal)까지 얼마나 움직여야 하는지 계산
                        l_dMi = self.robot.state.l_oMi.inverse() * self.l_goal
                        x_err_l = pin.log(l_dMi).vector # 6차원 벡터 [v, w]

                        # 2. Damped Least Squares (DLS) 적용
                        # np.linalg.pinv 대신 damping을 추가하여 특이점(팔이 다 펴진 상태 등)에서 발산을 방지
                        J_l = self.robot.state.l_J # Local Jacobian
                        damping = 1e-4
                        # J_inv = J.T * (J*J.T + damp*I)^-1
                        J_inv_l = J_l.T @ np.linalg.inv(J_l @ J_l.T + damping * np.eye(6))

                        # 3. 관절 속도 및 적분
                        kp = 1.0 # Gain
                        l_qdot = J_inv_l @ (kp * x_err_l)
                        
                        # pin.integrate를 사용하면 회전 관절의 수학적 한계를 정확히 처리합니다.
                        # self.l_qdes는 이전 루프의 q값 혹은 현재 q값에서 시작
                        dt = 0.01
                        self.l_qdes = pin.integrate(self.robot.model, 
                                                    np.append(self.l_qdes, np.zeros(7)), # 임시로 14차원 구성
                                                    np.append(l_qdot, np.zeros(7)) * dt)[:7]

                        # 4. 최종 명령 구성 (오른손은 초기 상태 혹은 0으로 고정)
                        qdes[:7] = self.l_qdes
                        qdes[7:] = np.zeros(7) # 오른손 고정

                        '''
                        # self.get_logger().info(f"self.l_goal {self.l_goal}...")
                        # self.get_logger().info(f"self.r_goal {self.r_goal}...")
                        # self.get_logger().info(f"self.l_qdes {self.l_qdes}...")
                        # self.get_logger().info(f"self.r_qdes {self.r_qdes}...")
                        # 1. 포즈 오차 계산 (목표와 현재의 차이)
                        l_dMi = self.robot.state.l_oMi.inverse() * self.l_goal
                        r_dMi = self.robot.state.r_oMi.inverse() * self.r_goal

                        x_err_l = pin.log(l_dMi).vector
                        x_err_r = pin.log(r_dMi).vector
                        # print("x_err_l:", x_err_l)
                        # 2. 기본 공식 (Primary Task만 적용)
                        # qdot = J_inv * (Gain * error)
                        l_qdot = np.linalg.pinv(self.robot.state.l_J) @ (1 * x_err_l)
                        r_qdot = np.linalg.pinv(self.robot.state.r_J) @ (1 * x_err_r)

                        # 3. 적분하여 명령 생성
                        # print("l_qdes:", self.l_qdes)
                        # print("l_qdot:", l_qdot)
                        self.l_qdes += l_qdot * 0.01
                        self.r_qdes += r_qdot * 0.01

                        qdes = np.append(self.l_qdes, self.r_qdes)
                        '''

                # print("Desired joint positions:", qdes)
        # qdes[7:] = np.zeros(7)  # 오른손 고정
        # qdes[:7] = np.array([-0.00445157, 0.00026862,0.00076013,-0.04703118,-0.00026862,0.05133107, 0.00039203])
        # qdes = np.zeros((14))
        self.step += 1
        # qdes[6] = 1.57-0.01*self.step
        # qdes[13] = 1.57+0.01*self.step
        # self.get_logger().info(f"Current self.r_goal {self.r_goal}...")
        # self.get_logger().info(f"Current self.l_goal {self.l_goal}...")
        # self.get_logger().info(f"Current self.l_qdes {self.l_qdes}...")
        # self.get_logger().info(f"Current self.l_oMi {self.robot.state.l_oMi}...")
        # self.get_logger().info(f"Current self.r_qdes {self.r_qdes}...")
        self.arm_publish(qdes)
        # self.arm_publish_test(np.zeros((14)))

    '''
    Mujoco JointState: 현재 robot state 정보를 받아서 H12Wrapper의 state에 반영
    '''
    def arm_publish(self, qdes=None):
        arm_msg = Float64MultiArray()
        arm_msg.data = qdes.tolist()
        # arm_msg.data = self.robot.state.q.tolist()
        # print("Publishing desired joint positions:", arm_msg.data)
        self.publisher.publish(arm_msg)

    def arm_publish_test(self, qdes=None):
        arm_msg = Float64MultiArray()
        qdes[7:] = np.zeros(7)  # 오른손 고정
        # qdes[:7] = np.array([-0.00445157, 0.00026862,0.00076013,-0.04703118,-0.00026862,0.05133107, 0.00039203])
        # qdes[7:] = np.zeros(7)  # 오른손 고정
        # arm_msg.data = qdes.tolist()
        
        if self.step==1:
            qdes[:7] = np.array([-0.00445157, 0.00026862,0.00076013,-0.04703118,-0.00026862,0.05133107, 0.00039203])
            print("1번째 state.l_oMi:", self.robot.state.l_oMi) 

        #     arm_msg.data = qdes.tolist()
        #     # arm_msg.data = self.robot.state.q.tolist()
        #     # print("Publishing desired joint positions:", arm_msg.data)
        else:
            qdes[:7] = np.array([-0.00890315,0.00053724,0.00152026,-0.09406236,-0.00053724,0.10266214,0.00078405])
            print("2번째 state.l_oMi:", self.robot.state.l_oMi) 
        arm_msg.data = qdes.tolist()
        self.get_logger().info(f"arm_msg {arm_msg}...")
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
    node = TeleopArmNodeTest()
    rclpy.spin(node)
    # node.main()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
