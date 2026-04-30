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

THRES = 0.1

class TeleopArmIgrisCRealNode(Node):
    def __init__(self):
        super().__init__('teleop_arm_igris_c_real_node')
        self.get_logger().info('Teleop Arm IgrisC Real Node has been started.')

        self.head_goal = pin.SE3()
        self.l_goal = pin.SE3()
        self.r_goal = pin.SE3()
        self.l_qdes = np.zeros(7)
        self.r_qdes = np.zeros(7)
        self.lfingers_goal = []
        self.rfingers_goal = []
        self.lfingers_qdes = np.zeros(6)
        self.rfingers_qdes = np.zeros(6)
        self.test_ = np.zeros(14)

        self.robot = IgrisCWrapper()

        # IgrisC State Subscriber
        self.subscriber = self.create_subscription(JointState, '/real/igrisc/arm/joint_states', self.joint_state_callback, 10)
        # IgrisC Command Publisher
        self.publisher = self.create_publisher(JointState, '/real/igrisc/arm/command', 10)

        # AVP Subscribers
        self.head_subscriber = self.create_subscription(Pose, '/head', self.head_callback, 10)
        self.lwrist_subscriber = self.create_subscription(Pose, '/lwrist', self.lwrist_callback, 10)
        self.rwrist_subscriber = self.create_subscription(Pose, '/rwrist', self.rwrist_callback, 10)

        # AVP Flag
        self.lwrist_ctrlFlag = False    
        self.rwrist_ctrlFlag = False
        self.head_ctrlFlag = False

        self.dt = 0.00333
        self.timer = self.create_timer(self.dt, self.timer_callback)

        self.ctrlFlag = False
        self.initCtrlFlag = True        # teleop 시작 전 초기화 플래그

        self.lwrist_rot_diff = np.array([[1,0,0],[0,0,1],[0,-1,0]]).T @ np.array([[0,0,-1],[0,1,0],[1,0,0]])
        self.rwrist_rot_diff = np.array([[1,0,0],[0,0,-1],[0,1,0]]).T @ np.array([[0,0,-1],[0,1,0],[1,0,0]])
        self.lwrist_rot_diff_se3 = pin.SE3(self.lwrist_rot_diff, np.zeros(3))
        self.rwrist_rot_diff_se3 = pin.SE3(self.rwrist_rot_diff, np.zeros(3))

    def timer_callback(self):
        qdes = np.zeros(14)
        if self.ctrlFlag:
            if self.head_ctrlFlag and self.lwrist_ctrlFlag and self.rwrist_ctrlFlag:
                # ROBOT
                headMlwrist = self.robot.state.head_oMi.inverse() * self.robot.state.l_oMi
                headMrwrist = self.robot.state.head_oMi.inverse() * self.robot.state.r_oMi
                
                # USER
                headMl_target =  self.head_goal.inverse() * self.l_goal * self.lwrist_rot_diff_se3
                headMr_target =  self.head_goal.inverse() * self.r_goal * self.rwrist_rot_diff_se3
                
                headMl_target.translation = headMl_target.translation.copy() * 1.0  #+ np.array([0,0,0.05])
                headMr_target.translation = headMr_target.translation.copy() * 1.0 #+ np.array([0,0,0.05])

                l_dMi = headMlwrist.inverse() * headMl_target
                r_dMi = headMrwrist.inverse() * headMr_target

                x_err_l = pin.log(l_dMi).vector
                x_err_r = pin.log(r_dMi).vector

                lambda_val = 0.01
                Kp = 0.1
                # 2. Left Hand DLS Inverse
                l_J = self.robot.state.l_J
                l_Identity = np.eye(l_J.shape[0])  # Task space 차원 (보통 6x6)
                # DLS 수식: J^T * (J * J^T + lambda^2 * I)^-1
                l_J_dls = l_J.T @ np.linalg.inv(l_J @ l_J.T + (lambda_val**2) * l_Identity)
                l_qdot = l_J_dls @ (Kp * x_err_l)

                # 3. Right Hand DLS Inverse
                r_J = self.robot.state.r_J
                r_Identity = np.eye(r_J.shape[0])
                r_J_dls = r_J.T @ np.linalg.inv(r_J @ r_J.T + (lambda_val**2) * r_Identity)
                r_qdot = r_J_dls @ (Kp * x_err_r)

                # qdes = self.robot.state.q.copy()
                # qdes[:7] += l_qdot * self.dt
                # qdes[7:] += r_qdot * self.dt

                self.l_qdes += l_qdot * 0.01
                self.r_qdes += r_qdot * 0.01
                qdes = np.append(self.l_qdes, self.r_qdes)

                self.arm_publish(qdes)
        # self.test_publish(qdes)

    def test_publish(self, qdes):
        arm_msg = JointState()
        qdes = self.test_.copy()
        qdes[-1] = 0.0
        # print("qdesL ", qdes)
        arm_msg.position = qdes.tolist()
        self.publisher.publish(arm_msg)

    def arm_publish(self, qdes):
        arm_msg = JointState()
        arm_msg.position = qdes.tolist()
        # Threshold 이용
        # print("arm publish qdesL ", qdes)
        # if abs(self.test_ - qdes).max() > THRES:
        #     print("값이 큼!! ")
        #     print("qdesL ", qdes)
        #     return
        self.publisher.publish(arm_msg)
    
    def flag_publish(self, flag):
        flag_msg = Bool()
        flag_msg.data = not flag
        self.flag_publisher.publish(flag_msg)

    def joint_state_callback(self, msg: JointState):
        self.test_ = np.array(msg.position)
        self.robot.state.q = np.array(msg.position) # 14
        self.robot.state.v = np.array(msg.velocity) # 14
        self.robot.computeAllTerms()

        if self.ctrlFlag == False:
            self.l_qdes = self.robot.state.q[:7].copy()
            self.r_qdes = self.robot.state.q[7:].copy()

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

def main():
    rclpy.init()
    node = TeleopArmIgrisCRealNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
