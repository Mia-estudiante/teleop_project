import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, PoseArray

import pinocchio as pin
import numpy as np
import os

# dex_retargeting 및 관련 유틸리티
from teleop_manager.src.dex_retargeting.DexRetargeting import DexRetargeting
# from dex_retargeting.retargeting_config import RetargetingConfig
from ament_index_python.packages import get_package_share_directory

class TeleopIgrisCRvizNode(Node):
    def __init__(self):
        super().__init__('teleop_igris_c_rviz_node')
        self.get_logger().info('Teleop IgrisC RViz Node started.')

        # 1. 시각화용 JointState 발행자
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)
        
        # 2. 통합 URDF 조인트 이름 (제공해주신 RViz 53개 리스트 순서와 100% 일치) 
        self.all_joint_names = [
            "0_Joint_Waist_Yaw", "1_Joint_Waist_Roll", "2_Joint_Waist_Pitch",
            "3_Joint_Hip_Pitch_Left", "4_Joint_Hip_Roll_Left", "5_Joint_Hip_Yaw_Left",
            "6_Joint_Knee_Pitch_Left", "7_Joint_Ankle_Pitch_Left", "8_Joint_Ankle_Roll_Left",
            "9_Joint_Hip_Pitch_Right", "10_Joint_Hip_Roll_Right", "11_Joint_Hip_Yaw_Right",
            "12_Joint_Knee_Pitch_Right", "13_Joint_Ankle_Pitch_Right", "14_Joint_Ankle_Roll_Right",
            "15_Joint_Shoulder_Pitch_Left", "16_Joint_Shoulder_Roll_Left", "17_Joint_Shoulder_Yaw_Left",
            "18_Joint_Elbow_Pitch_Left", "19_Joint_Wrist_Yaw_Left", "20_Joint_Wrist_Roll_Left", "21_Joint_Wrist_Pitch_Left",
            "Left_0_Joint_Thumb_Proximal", "Left_1_Joint_Thumb_Middle", "Left_2_Joint_Thumb_Distal",
            "Left_3_Joint_Index_Middle", "Left_4_Joint_Index_Distal", "Left_5_Joint_Middle_Middle",
            "Left_6_Joint_Middle_Distal", "Left_7_Joint_Ring_Middle", "Left_8_Joint_Ring_Distal",
            "Left_9_Joint_Little_Middle", "Left_10_Joint_Little_Distal",
            "22_Joint_Shoulder_Pitch_Right", "23_Joint_Shoulder_Roll_Right", "24_Joint_Shoulder_Yaw_Right",
            "25_Joint_Elbow_Pitch_Right", "26_Joint_Wrist_Yaw_Right", "27_Joint_Wrist_Roll_Right", "28_Joint_Wrist_Pitch_Right",
            "Right_0_Joint_Thumb_Proximal", "Right_1_Joint_Thumb_Middle", "Right_2_Joint_Thumb_Distal",
            "Right_3_Joint_Index_Middle", "Right_4_Joint_Index_Distal", "Right_5_Joint_Middle_Middle",
            "Right_6_Joint_Middle_Distal", "Right_7_Joint_Ring_Middle", "Right_8_Joint_Ring_Distal",
            "Right_9_Joint_Little_Middle", "Right_10_Joint_Little_Distal",
            "29_Joint_Neck_Yaw", "30_Joint_Neck_Pitch"
        ]

        # 3. 리타겟팅 설정 로드
        self.dexretargeting = DexRetargeting('igris_c')

        # 4. 구독자 설정 (AVP Pose 데이터)
        self.l_wrist_pose = pin.SE3.Identity()
        self.r_wrist_pose = pin.SE3.Identity()
        self.lfingers_poses = []
        self.rfingers_poses = []
        
        self.create_subscription(Pose, '/lwrist', self.lwrist_callback, 10)
        self.create_subscription(Pose, '/rwrist', self.rwrist_callback, 10)
        self.create_subscription(PoseArray, '/lfingers', self.lfingers_callback, 10)
        self.create_subscription(PoseArray, '/rfingers', self.rfingers_callback, 10)

        # 5. 상태 변수 및 타이머
        self.q_current = [0.0] * len(self.all_joint_names)
        self.timer = self.create_timer(0.01, self.timer_callback) # 100Hz

    def lwrist_callback(self, msg):
        self.l_wrist_pose = pin.XYZQUATToSE3([msg.position.x, msg.position.y, msg.position.z, 
                                              msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])

    def rwrist_callback(self, msg):
        self.r_wrist_pose = pin.XYZQUATToSE3([msg.position.x, msg.position.y, msg.position.z, 
                                              msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])

    def lfingers_callback(self, msg):
        self.lfingers_poses = [pin.XYZQUATToSE3([p.position.x, p.position.y, p.position.z, 
                                                 p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]) for p in msg.poses]
    def rfingers_callback(self, msg):
        self.rfingers_poses = [pin.XYZQUATToSE3([p.position.x, p.position.y, p.position.z, 
                                                 p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]) for p in msg.poses]

    def timer_callback(self):
        if not self.lfingers_poses or not self.rfingers_poses:
            return

        # 1. 손목 기준 상대 좌표 변환 (Retargeting용)
        l_ref = [(self.l_wrist_pose.inverse() * f).translation for f in self.lfingers_poses]
        r_ref = [(self.r_wrist_pose.inverse() * f).translation for f in self.rfingers_poses]
        # 2. Retargeting 실행
        # print("self.left_retargeting.optimizer.target_joint_names:", self.dexretargeting.left_retargeting.optimizer.target_joint_names)
        l_qpos = self.dexretargeting.retarget_ref(
            self.dexretargeting.left_retargeting, 
            np.array(l_ref)
        )[self.dexretargeting.left_retargeting_index]

        # print("self.dexretargeting.right_retargeting_index:", self.dexretargeting.right_retargeting_index)
        r_qpos = self.dexretargeting.retarget_ref(
            self.dexretargeting.right_retargeting, 
            np.array(r_ref)
        )[self.dexretargeting.right_retargeting_index]

        # 3. 전체 JointState 메시지 생성
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.all_joint_names
        
        # 기본 위치값 0.0 (허리, 다리 등 고정)
        q_out = [0.0] * len(self.all_joint_names)

        # 리타겟팅된 조인트 값을 전체 배열에 매핑
        def map_to_all(side, qpos, joint_names):
            for i, name in enumerate(joint_names):
                if side+name in self.all_joint_names:
                    q_out[self.all_joint_names.index(side+name)] = float(qpos[i])
                    print("Mapping:", side+name, "->", self.all_joint_names.index(side+name))
        map_to_all('Left_', l_qpos, self.dexretargeting.left_retargeting.optimizer.target_joint_names)
        map_to_all('Right_', r_qpos, self.dexretargeting.right_retargeting.optimizer.target_joint_names)

        # 4. Mimic 조인트 수동 보정 (Distal 마디 굽히기) 
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
            q_out[self.all_joint_names.index(child)] = q_out[p_idx]

        msg.position = q_out
        print("Publishing JointState:", msg.position)
        self.joint_pub.publish(msg)

def main():
    rclpy.init()
    node = TeleopIgrisCRvizNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()