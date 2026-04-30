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

class TeleopHandIgrisCNode(Node):
    def __init__(self):
        super().__init__('teleop_hand_igris_c_node')
        self.get_logger().info('Teleop Hand IgrisC Node has been started.')

        self.head_goal = pin.SE3()
        self.l_goal = pin.SE3()
        self.r_goal = pin.SE3()
        self.lfingers_goal = []
        self.rfingers_goal = []
        self.lfingers_qdes = np.zeros(6)
        self.rfingers_qdes = np.zeros(6)
        # self.init_qdes = np.zeros(22)

        # self.robot = IgrisCWrapper()
        self.dexretargeting = DexRetargeting('igris_c')

        self.publisher = self.create_publisher(Float64MultiArray, '/unity/controller', 10)
        self.flag_publisher = self.create_publisher(Bool, '/unity/flag', 10)
        self.subscriber = self.create_subscription(Float64MultiArray, '/unity/hand_joint_states', self.hand_joint_state_callback, 10)
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

    def timer_callback(self):
        # qdes = np.zeros(26)
        qdes = np.zeros(12)
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
                        self.initCtrlFlag2 = False
                    else:
                        l_wristMl_fingers = []
                        r_wristMr_fingers = []
                        
                        for lfinger in self.lfingers_goal:
                            wristMfingers = (self.l_goal).inverse() * lfinger
                            l_wristMl_fingers.append(wristMfingers.translation.tolist())
                        for rfinger in self.rfingers_goal:
                            wristMfingers = (self.r_goal).inverse() * rfinger
                            r_wristMr_fingers.append(wristMfingers.translation.tolist())

                        self.lfingers_qdes = self.dexretargeting.retarget_ref(
                            self.dexretargeting.left_retargeting, 
                            np.array(l_wristMl_fingers)
                        )[self.dexretargeting.left_retargeting_index]

                        self.rfingers_qdes = self.dexretargeting.retarget_ref(
                            self.dexretargeting.right_retargeting, 
                            np.array(r_wristMr_fingers)
                        )[self.dexretargeting.right_retargeting_index]
                qdes = np.append(self.lfingers_qdes, self.rfingers_qdes)
        self.hand_publish(qdes)
        self.flag_publish(self.initCtrlFlag)

    def hand_publish(self, qdes):
        hand_msg = Float64MultiArray()
        hand_msg.data = qdes.tolist()
        self.publisher.publish(hand_msg)

    '''
    Mujoco JointState: 현재 robot state 정보를 받아서 H12Wrapper의 state에 반영
    '''
    # def arm_publish(self, qdes):
    #     arm_msg = Float64MultiArray()
    #     arm_msg.data = qdes.tolist()
    #     self.publisher.publish(arm_msg)
    
    def flag_publish(self, flag):
        flag_msg = Bool()
        flag_msg.data = not flag
        self.flag_publisher.publish(flag_msg)

    def hand_joint_state_callback(self, msg: Float64MultiArray):
        self.lfingers_qdes = np.array(msg.data[:6])
        self.rfingers_qdes = np.array(msg.data[6:])
        # self.robot.state.q = np.array(msg.position) # 14
        # self.robot.state.v = np.array(msg.velocity) # 14
        # self.robot.computeAllTerms()
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
    node = TeleopHandIgrisCNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
