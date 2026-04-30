import os

import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, PoseArray

# pinocchio
import pinocchio as pin
from pinocchio.utils import *

import json
import numpy as np

from teleop_manager.src.robot.h1_2_wrapper import H12Wrapper
from teleop_manager.src.dex_retargeting.DexRetargeting import DexRetargeting

THRES = 3e-4

class TeleopArmHandNode(Node):
    def __init__(self):
        super().__init__('teleop_arm_hand_node')
        self.get_logger().info('Teleop Arm Hand Node has been started.')

        # self.head_goal = pin.SE3()
        self.l_goal = pin.SE3()
        self.r_goal = pin.SE3()
        self.lfingers_goal = np.zeros(12)
        self.rfingers_goal = np.zeros(12)
        self.lfingers_qdes = np.zeros(12)
        self.rfingers_qdes = np.zeros(12)

        self.robot = H12Wrapper()
        self.dexretargeting = DexRetargeting()
        self.left_retargeting_to_mjc = np.array([self.robot.joint_names.index('L_'+name) for name in self.dexretargeting.left_retargeting_joint_names if 'L_'+name in self.robot.joint_names]).astype(int)
        self.right_retargeting_to_mjc = np.array([self.robot.joint_names.index('R_'+name) for name in self.dexretargeting.right_retargeting_joint_names if 'R_'+name in self.robot.joint_names]).astype(int)

        self.ref_value_json = '/home/home/mujoco_ws/src/user_manager/user_manager/src/data/mediapipe_ref.json'
        self.mujoco_qpos_json = '/home/home/mujoco_ws/src/user_manager/user_manager/src/data/mujoco_qpos.json'
        self.mujoco_qpos = []

        with open(self.ref_value_json, 'r') as f:
            self.ref_data = json.load(f)
        self.ref_data_idx = 0 # 현재 프레임 위치 추적용
        self.total_ref_frames = len(self.ref_data)
        
        self.publisher = self.create_publisher(Float64MultiArray, '/mujoco/controller', 10)
        self.subscriber = self.create_subscription(JointState, '/mujoco/joint_states', self.joint_state_callback, 10)
        # self.head_subscriber = self.create_subscription(Pose, '/head', self.head_callback, 10)
        self.lwrist_subscriber = self.create_subscription(Pose, '/lwrist', self.lwrist_callback, 10)
        self.rwrist_subscriber = self.create_subscription(Pose, '/rwrist', self.rwrist_callback, 10)
        self.lfingers_subscriber = self.create_subscription(PoseArray, '/lfingers', self.lfingers_callback, 10)
        self.rfingers_subscriber = self.create_subscription(PoseArray, '/rfingers', self.rfingers_callback, 10)

        self.timer = self.create_timer(0.01, self.timer_callback)
        
        self.ctrlFlag = False           
        self.initCtrlFlag = True        # teleop 시작 전 초기화 플래그
        self.initCtrlFlag2 = True       # teleop 시작 전 초기화 플래그
        ############
        self.lwrist_ctrlFlag = True    
        self.rwrist_ctrlFlag = True     
        self.lfingers_ctrlFlag = True     
        self.rfingers_ctrlFlag = True
        ############
        # self.head_ctrlFlag = False

    def timer_callback(self):
        qdes = np.zeros(38)
        if self.ctrlFlag:
            if self.initCtrlFlag:
                self.get_logger().info(f"Initializing...")
                self.get_logger().info(f"Initializing Done. Moving to teleoperation...")
                self.initCtrlFlag = False
            else:
                if self.lwrist_ctrlFlag and self.rwrist_ctrlFlag and self.lfingers_ctrlFlag and self.rfingers_ctrlFlag:
                    if self.initCtrlFlag2:
                        self.lfingers_qdes = self.robot.state.q[14:26].copy()
                        self.rfingers_qdes = self.robot.state.q[33:45].copy()
                        self.initCtrlFlag2 = False
                    else:
                        current_frame_data = self.ref_data[self.ref_data_idx]['data']
                        ref_value = np.array(current_frame_data)
                        self.lfingers_qdes = self.dexretargeting.retarget_ref(self.dexretargeting.left_retargeting, ref_value)
                        self.mujoco_qpos.append(self.lfingers_qdes.tolist())
                        
                        # self.lfingers_qdes = self.dexretargeting.retarget(self.dexretargeting.left_retargeting, self.dexretargeting.left_indices, self.lfingers_goal)
                        # self.rfingers_qdes = self.dexretargeting.retarget(self.dexretargeting.right_retargeting, self.dexretargeting.right_indices, self.rfingers_goal)
                        # print("self.lfingers_qdes", self.lfingers_qdes)
                        # print("self.rfingers_qdes", self.rfingers_qdes.shape)
                        self.ref_data_idx = (self.ref_data_idx + 1) % self.total_ref_frames
                    # # Left Hand Retargeting
                    # left_joint_pos = self.robot.state.q[self.left_retargeting_to_mjc]
                    # left_qpos = self.dexretargeting.left_retargeting.retarget(left_joint_pos)
                    # # Right Hand Retargeting
                    # right_joint_pos = self.robot.state.q[self.right_retargeting_to_mjc]
                    # right_qpos = self.dexretargeting.right_retargeting.retarget(right_joint_pos)

                    # # Combine Left and Right Hand qdes
                    # for i, idx in enumerate(self.left_retargeting_to_mjc):
                    #     qdes[idx] = left_qpos[i]
                    # for i, idx in enumerate(self.right_retargeting_to_mjc):
                    #     qdes[idx] = right_qpos[i]
        qdes[:7] = 0 # left wrist
        qdes[7:19] = self.lfingers_qdes # left fingers
        qdes[19:26] = 0 # right wrist
        # print("qdes[26:]", qdes[26:])
        # 수정 전: qdes[26:] = self.rfingers_qdes

        # 수정 후: 데이터 개수가 맞을 때만 할당하도록 변경
        # if len(self.rfingers_qdes) == 12:
        #     qdes[26:] = self.rfingers_qdes
        # else:
        #     self.get_logger().warn(
        #         f"데이터 크기 불일치! 예상: 12, 실제: {len(self.rfingers_qdes)}. "
        #         f"데이터 내용: {self.rfingers_qdes}"
        #     )        # print(qdes.shape)
        # print("self.rfingers_qdes", self.rfingers_qdes)
        # print("qdes[26:]", qdes[26:])
        qdes[26:] = 0 # right fingers

        # print(qdes)
        # qdes[7:14] = 0 # left wrist
        # qdes[14:26] = self.lfingers_qdes # left fingers
        # qdes[26:33] = 0 # right wrist
        # qdes[33:45] = self.rfingers_qdes # right fingers
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

    # def head_callback(self, msg: Pose):
    #     pos = np.array([msg.position.x, msg.position.y, msg.position.z, msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
    #     self.head_goal = pin.XYZQUATToSE3(pos)
    #     self.head_ctrlFlag = True

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
            lfingers_goal.append([msg.poses[i].position.x, msg.poses[i].position.y, msg.poses[i].position.z])
            # pos_list.append([msg.poses[i].position.x, msg.poses[i].position.y, msg.poses[i].position.z, msg.poses[i].orientation.x, msg.poses[i].orientation.y, msg.poses[i].orientation.z, msg.poses[i].orientation.w])
        self.lfingers_goal = np.array(lfingers_goal)
        # print("lfingers_callback pos_list:", pos_list)
        # self.l_goal = pin.XYZQUATToSE3(pos)
        self.lfingers_ctrlFlag = True

    def rfingers_callback(self, msg: PoseArray):
        rfingers_goal = []
        for i in range(len(msg.poses)):
            rfingers_goal.append([msg.poses[i].position.x, msg.poses[i].position.y, msg.poses[i].position.z])
            # pos_list.append([msg.poses[i].position.x, msg.poses[i].position.y, msg.poses[i].position.z, msg.poses[i].orientation.x, msg.poses[i].orientation.y, msg.poses[i].orientation.z, msg.poses[i].orientation.w])
        self.rfingers_goal = np.array(rfingers_goal)
        # print("rfingers_callback pos_list:", pos_list)
        # self.r_goal = pin.XYZQUATToSE3(pos)
        self.rfingers_ctrlFlag = True   

    # 3. 종료 시 호출할 저장 함수
    def save_qdes_data(self):
        if not self.mujoco_qpos:
            self.get_logger().warn("저장할 데이터가 없습니다.")
            return
        try:
            with open(self.mujoco_qpos_json, 'w') as f:
                json.dump(self.mujoco_qpos, f)
            self.get_logger().info(f"데이터 저장 완료: {self.mujoco_qpos_json} (총 {len(self.mujoco_qpos)} 프레임)")
        except Exception as e:
            self.get_logger().error(f"데이터 저장 실패: {str(e)}")

# def main():
#     rclpy.init()
#     node = TeleopArmHandNode()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()
def main():
    rclpy.init()
    node = TeleopArmHandNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT) received.')
    finally:
        # 노드 종료 전 데이터 저장
        node.save_qdes_data()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
