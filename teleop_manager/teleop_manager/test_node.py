from mimetypes import init
import rclpy
from rclpy.node import Node

import numpy as np

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from teleop_manager.src.robot.igris_c_wrapper import IgrisCWrapper
import pinocchio as pin
from pinocchio.utils import *

class Trajectory:
    def __init__(self):
        self.points = []
    def cubic_spline_vector(self, q_init, q_des, T, dt):
        """
        q_init: 시작 위치 (np.array, shape: (n,))
        q_des: 목표 위치 (np.array, shape: (n,))
        T: 이동 시간
        dt: 샘플링 주기
        """
        t = np.arange(0, T + dt, dt).reshape(-1, 1) # (N, 1) 형태로 변환하여 브로드캐스팅 준비
        
        # 계수 계산 (벡터 연산)
        a0 = q_init
        a1 = np.zeros_like(q_init)
        a2 = (3 / T**2) * (q_des - q_init)
        a3 = (-2 / T**3) * (q_des - q_init)
        
        # 위치 계산: t (N, 1)와 계수 (n,)의 연산 결과는 (N, n) 행렬
        # 각 열은 각 관절의 궤적이 됩니다.
        q_t = a0 + a1*t + a2*t**2 + a3*t**3
        v_t = a1 + 2*a2*t + 3*a3*t**2
        
        return t.flatten(), q_t, v_t

class TestNode(Node):
    def __init__(self):
        super().__init__('test_node')
        self.get_logger().info('Test Node has been started.')

        self.q_des = np.zeros((14))
        self.q_des[3] = self.q_des[10] = -np.pi/2
        self.q_tmp = np.zeros((14))
        self.time = None

        self.trajectory = Trajectory()

        self.joint_state_sub = self.create_subscription(JointState, '/real/igrisc/arm/joint_states', self.joint_state_callback, 10)
        # self.joint_state_sub = self.create_subscription(JointState, '/mujoco/joint_states', self.joint_state_callback, 10)
        self.publisher = self.create_publisher(JointState, '/real/igrisc/arm/command', 10)
        # self.publisher = self.create_publisher(Float64MultiArray, '/mujoco/controller', 10)
        
        # State
        self.q_current = np.zeros((14))
        self.v_current = np.zeros((14))

        # Timer
        self.timer = self.create_timer(0.00333, self.timer_callback)

        # Initial
        self.init = True
    
        self.iter = 0
        self.joint_state_ok = False

        self.robot = IgrisCWrapper()

        self.mode = 1
        self.target_l = None
        self.target_r = None

        self.a = +1.0



    def joint_state_callback(self, msg):
        self.q_current = np.array(msg.position)
        self.v_current = np.array(msg.velocity)
        # print(f"Current joint states: {self.q_current}")
        self.joint_state_ok = True

        self.robot.state.q = self.q_current.copy()
        self.robot.state.v = self.v_current.copy()
        self.robot.computeAllTerms()

        # print(f"L Link positions: {self.robot.state.l_oMi}")
        # print(f"R Link positions: {self.robot.state.r_oMi}")

    def timer_callback(self):
        qdes = np.zeros(14)
        msg = JointState()
        msg.position = qdes.tolist()
        self.publisher.publish(msg) 
        self.get_logger().info(f"Published zero command: {qdes}")   



    # def timer_callback(self):
    #     if self.joint_state_ok:
    #         if self.mode == 1:
    #             T = 3.0
    #             dt = 0.00333
    #             if self.init:
    #                 self.time, self.q_tmp, v_t = self.trajectory.cubic_spline_vector(self.q_current, self.q_des, T, dt)
    #                 self.init = False

    #             qdes = self.q_tmp[self.iter]
    #             self.iter += 1
    #             if self.iter >= len(self.time):
    #                 self.mode = 2
    #                 self.init = True
    #                 self.iter = 0
    #         elif self.mode == 2:
    #             if self.init:
    #                 self.get_logger().info(f"Mode Change")
    #                 self.get_logger().info(f"L_oMI: {self.robot.state.l_oMi}")
    #                 self.get_logger().info(f"R_oMI: {self.robot.state.r_oMi}")
    #                 self.target_l = self.robot.state.l_oMi.copy()
    #                 self.target_r = self.robot.state.r_oMi.copy()
    #                 self.init = False
    #             self.target_l.translation[1] += self.a * 0.0001
    #             self.target_r.translation[1] -= self.a *0.0001

    #             dMi_l = self.robot.state.l_oMi.inverse() * self.target_l
    #             dMi_r = self.robot.state.r_oMi.inverse() * self.target_r
    #             x_err_l = pin.log(dMi_l).vector
    #             x_err_r = pin.log(dMi_r).vector

    #             Kp = 100.0
    #             # 1. Damping Factor 설정 (보통 0.01 ~ 0.1 사이에서 튜닝)
    #             lambda_val = 0.01

    #             # 2. Left Hand DLS Inverse
    #             l_J = self.robot.state.l_J
    #             l_Identity = np.eye(l_J.shape[0])  # Task space 차원 (보통 6x6)
    #             # DLS 수식: J^T * (J * J^T + lambda^2 * I)^-1
    #             l_J_dls = l_J.T @ np.linalg.inv(l_J @ l_J.T + (lambda_val**2) * l_Identity)
    #             l_qdot = l_J_dls @ (Kp * x_err_l)

    #             # 3. Right Hand DLS Inverse
    #             r_J = self.robot.state.r_J
    #             r_Identity = np.eye(r_J.shape[0])
    #             r_J_dls = r_J.T @ np.linalg.inv(r_J @ r_J.T + (lambda_val**2) * r_Identity)
    #             r_qdot = r_J_dls @ (Kp * x_err_r)

    #             qdes = self.q_current.copy()
    #             qdes[:7] += l_qdot * 0.00333
    #             qdes[7:] += r_qdot * 0.00333
    #             # print(f"qdes: {qdes}")  

    #             self.iter +=1   
    #             if self.iter >= 1000:
    #                 self.get_logger().info(f"Toggle")
    #                 self.get_logger().info(f"L_oMI: {self.robot.state.l_oMi}")
    #                 self.get_logger().info(f"R_oMI: {self.robot.state.r_oMi}")
    #                 self.get_logger().info(f"target_l: {self.target_l}")
    #                 self.get_logger().info(f"target_r: {self.target_r}")
    #                 self.a *= -1.0
    #                 self.iter = 0

    #         msg = JointState()
    #         msg.position = qdes.tolist()

    #         # msg = Float64MultiArray()
    #         # msg.data = qdes.tolist()
    #         self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = TestNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()