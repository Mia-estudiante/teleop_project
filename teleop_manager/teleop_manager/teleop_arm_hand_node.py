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
from teleop_manager.src.dex_retargeting.DexRetargeting import DexRetargeting
from simulation.simulation_hand_node import MujocoSimulationHandNode

THRES = 3e-4

class TeleopArmHandNode(Node):
    def __init__(self):
        super().__init__('teleop_arm_hand_node')
        self.get_logger().info('Teleop Arm Hand Node has been started.')

        # self.head_goal = pin.SE3()
        self.l_goal = pin.SE3()
        self.r_goal = pin.SE3()
        self.lfingers_goal = []
        self.rfingers_goal = []
        self.lfingers_qdes = np.zeros(6)
        self.rfingers_qdes = np.zeros(6)
        self.init_qdes = np.zeros(38)

        self.robot = H12Wrapper()
        self.dexretargeting = DexRetargeting()

        self.left_retargeting_to_mjc = MujocoSimulationHandNode().left_retargeting_to_mjc
        self.right_retargeting_to_mjc = MujocoSimulationHandNode().right_retargeting_to_mjc

        # self.mujoco_qpos = []

        self.publisher = self.create_publisher(Float64MultiArray, '/mujoco/controller', 10)
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
        ############
        self.head_ctrlFlag = False    
        self.lwrist_ctrlFlag = False    
        self.rwrist_ctrlFlag = False     
        self.lfingers_ctrlFlag = False     
        self.rfingers_ctrlFlag = False
        ############

        self.lwrist_rot_diff = np.array([[1,0,0],[0,0,1],[0,-1,0]]).T
        self.rwrist_rot_diff = np.array([[1,0,0],[0,0,-1],[0,1,0]]).T
        self.lfingers_rot_diff_se3 = pin.SE3(self.lwrist_rot_diff, np.zeros(3))
        self.rfingers_rot_diff_se3 = pin.SE3(self.rwrist_rot_diff, np.zeros(3))
        
    def timer_callback(self):
        qdes = np.zeros(26)
        if self.ctrlFlag:
            if self.initCtrlFlag:
                self.get_logger().info(f"Initializing...")
                # if np.linalg.norm(self.robot.state.q - self.init_qdes) < THRES:
                #     qdes = self.init_qdes.copy()
                self.initCtrlFlag = False
                self.get_logger().info(f"Initializing Done. Moving to teleoperation...")
            else:
                if self.head_ctrlFlag and self.lwrist_ctrlFlag and self.rwrist_ctrlFlag and self.lfingers_ctrlFlag and self.rfingers_ctrlFlag:
                    if self.initCtrlFlag2:
                        # self.l_qdes = self.robot.state.q[:7].copy()
                        # self.r_qdes = self.robot.state.q[19:26].copy()
                        self.initCtrlFlag2 = False
                    else:
                        '''
                        # ROBOT
                        headMlwrist = self.robot.state.head_oMi.inverse() * self.robot.state.l_oMi
                        headMrwrist = self.robot.state.head_oMi.inverse() * self.robot.state.r_oMi
                        
                        # USER
                        headMl_target =  self.head_goal.inverse() * self.l_goal
                        headMr_target =  self.head_goal.inverse() * self.r_goal
                        
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
                        '''

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

        qdes[:7] = 0#self.l_qdes # left wrist
        qdes[7:13] = self.lfingers_qdes # left fingers
        qdes[13:20] = 0 #self.r_qdes # right wrist
        qdes[20:] = self.rfingers_qdes # right fingers
        self.hand_publish(qdes)

    '''
    Mujoco JointState: 현재 robot state 정보를 받아서 H12Wrapper의 state에 반영
    '''
    def hand_publish(self, qdes):
        hand_msg = Float64MultiArray()
        hand_msg.data = qdes.tolist()
        self.publisher.publish(hand_msg)

    def joint_state_callback(self, msg: JointState):
        self.robot.state.q = np.array(msg.position) # 38
        self.robot.state.v = np.array(msg.velocity) # 38
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
    rclpy.init()
    node = TeleopArmHandNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
