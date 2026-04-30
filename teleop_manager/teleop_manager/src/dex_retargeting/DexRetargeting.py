import os
import numpy as np

import pinocchio as pin
from pinocchio.utils import *

# dex_retargeting
from dex_retargeting.retargeting_config import RetargetingConfig

from ament_index_python.packages import get_package_share_directory

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

# 1. JointState를 발행하는 클래스 추가
class JointPublisher(Node):
    def __init__(self, save_path, robot_name):
        super().__init__('hand_joint_publisher')
        self.save_path = save_path
        self.publisher = self.create_publisher(JointState, 'joint_states', 10)
        if robot_name == 'h1_2':
            self.target_joint_names = [
                'pinky_proximal_joint', 
                'ring_proximal_joint', 
                'middle_proximal_joint', 
                'index_proximal_joint',
                'thumb_proximal_pitch_joint', 
                'thumb_proximal_yaw_joint'
            ]
        elif robot_name == 'igris_c':
            self.target_joint_names = [ '9_Joint_Little_Middle', '7_Joint_Ring_Middle', '5_Joint_Middle_Middle', '3_Joint_Index_Middle',
                                        '1_Joint_Thumb_Middle', '0_Joint_Thumb_Proximal' ]
        
        # # 저장 경로 설정 및 기존 파일 초기화 (새로 쓰기)
        # with open(self.save_path, 'w') as f:
        #     # 헤더(조인트 이름)를 파일 맨 위에 저장해두면 나중에 분석하기 좋습니다.
        #     f.write("# " + " ".join(self.target_joint_names) + "\n")

    def publish_joints(self, qpos, joint_names_from_retargeting):
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        
        ordered_qpos = []
        for target_name in self.target_joint_names:
            if target_name in joint_names_from_retargeting:
                idx = joint_names_from_retargeting.index(target_name)
                # print("idx target_name", idx, target_name)
                ordered_qpos.append(float(qpos[idx]))
            else:
                ordered_qpos.append(0.0)

        msg.name = self.target_joint_names
        msg.position = ordered_qpos
        
        self.publisher.publish(msg)

        # --- 파일 저장 로직 추가 ---
        # self.save_to_txt(ordered_qpos)

    def save_to_txt(self, qpos_list):
        """리스트 데이터를 한 줄로 파일에 저장"""
        try:
            with open(self.save_path, 'a') as f:  # 'a' 모드는 append (이어쓰기)
                # 공백으로 구분하여 저장 (예: 0.12 0.45 0.11 ...)
                line = " ".join(map(str, qpos_list))
                f.write(line + "\n")
        except Exception as e:
            self.get_logger().error(f"Failed to save joint data: {e}")

class DexRetargeting:
    def __init__(self, robot_name):
        # URDF file
        if robot_name == 'h1_2':
            descriptions_path = get_package_share_directory('h1_2_description')
            RetargetingConfig.set_default_urdf_dir(str(descriptions_path))
            left_yml_path = os.path.join(descriptions_path, 'inspire_hand/inspire_hand_left.yml')
            right_yml_path = os.path.join(descriptions_path, 'inspire_hand/inspire_hand_right.yml')
        elif robot_name == 'igris_c':
            descriptions_path = get_package_share_directory('igris_c_description')
            RetargetingConfig.set_default_urdf_dir(str(descriptions_path))
            left_yml_path = os.path.join(descriptions_path, 'hand/left_hand_igris_c.yml')
            right_yml_path = os.path.join(descriptions_path, 'hand/right_hand_igris_c.yml')
        else:
            raise ValueError(f"Unknown robot name: {robot_name}")

        self.left_retargeting = RetargetingConfig.load_from_file(left_yml_path).build()
        self.right_retargeting = RetargetingConfig.load_from_file(right_yml_path).build()

        self.left_retargeting_joint_names = self.left_retargeting.joint_names
        self.right_retargeting_joint_names = self.right_retargeting.joint_names
        
        # [4, 6, 2, 0, 9, 8]
        self.left_retargeting_index = [self.left_retargeting_joint_names.index(target_joint_name) for target_joint_name in self.left_retargeting.optimizer.target_joint_names if target_joint_name in self.left_retargeting_joint_names] 
        self.right_retargeting_index = [self.right_retargeting_joint_names.index(target_joint_name) for target_joint_name in self.right_retargeting.optimizer.target_joint_names if target_joint_name in self.right_retargeting_joint_names]
        # print("self.left_retargeting_index", self.left_retargeting_index)
        # print("self.right_retargeting_index", self.right_retargeting_index)
        '''
        optimizer? 에 정의되어 joint 들로, 총 12개가 뽑힌다
        여기서 우리는 yaml 파일에 정의한 target_joint_names 을 가지고 qdes 를 추출하며,
        이것을 통해 mimic joint 의 qdes 값을 정의한다
        ['index_proximal_joint', 'index_intermediate_joint', 'middle_proximal_joint', 'middle_intermediate_joint', 'pinky_proximal_joint', 'pinky_intermediate_joint', 'ring_proximal_joint', 'ring_intermediate_joint', 'thumb_proximal_yaw_joint', 'thumb_proximal_pitch_joint', 'thumb_intermediate_joint', 'thumb_distal_joint']
        ['0_Joint_Thumb_Proximal', '1_Joint_Thumb_Middle', '2_Joint_Thumb_Distal', '3_Joint_Index_Middle', '4_Joint_Index_Distal', '5_Joint_Middle_Middle', '6_Joint_Middle_Distal', '7_Joint_Ring_Middle', '8_Joint_Ring_Distal', '9_Joint_Little_Middle', '10_Joint_Little_Distal']
        '''

    def retarget(self, retargeting, indices, joint_pos: np.ndarray) -> np.ndarray:
        origin_indices = indices[0, :]  # [ 0, 0, 0, 0, 0 ],
        task_indices = indices[1, :]    # [ 4, 9, 14, 19, 24 ] 
        ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
        qpos = retargeting.retarget(ref_value)
        return qpos
    
    def retarget_ref(self, retargeting, ref_value) -> np.ndarray:
        qpos = retargeting.retarget(ref_value)
        return qpos

    def load_txt(self, path):
        data = []
        try:
            with open(path, 'r') as f:
                for line in f:
                    if line.strip() == '':
                        continue
                    vals = list(map(float, line.strip().split()))
                    data.append(vals)
        except FileNotFoundError:
            self.get_logger().error(f'File not found: {path}')
        return data

def main():
    # ROS 2 초기화
    rclpy.init()
    
    dex_retargeting = DexRetargeting('igris_c')
    lfingers_save_path = '/retargeted_joints_left_igris.txt'
    rfingers_save_path = '/retargeted_joints_right_igris.txt'
    lfingers_joint_pub = JointPublisher(lfingers_save_path, 'igris_c') # 발행자 객체 생성
    rfingers_joint_pub = JointPublisher(rfingers_save_path, 'igris_c') # 발행자 객체 생성

    lfingers_file_path = '/avp_lwrist_fingers.txt'
    lfingers_data = dex_retargeting.load_txt(lfingers_file_path)
    rfingers_file_path = '/avp_rwrist_fingers.txt'
    rfingers_data = dex_retargeting.load_txt(rfingers_file_path)

    lfingers_frame_idx = 0
    rfingers_frame_idx = 0
    n_lfingers = (len(lfingers_data[0]) - 7) // 7
    n_rfingers = (len(rfingers_data[0]) - 7) // 7

    try:
        while rclpy.ok():
            # 1. AVP 데이터 프레임 가져오기
            lfingers_vals = lfingers_data[lfingers_frame_idx]
            rfingers_vals = rfingers_data[rfingers_frame_idx]

            # 2. Wrist 기준으로 변환 (상대 좌표 계산)
            l_wrist = lfingers_vals[0:7]
            l_wrist_se3 = pin.XYZQUATToSE3(np.array(l_wrist))
            r_wrist = rfingers_vals[0:7]
            r_wrist_se3 = pin.XYZQUATToSE3(np.array(r_wrist))

            l_wrist_To_fingers = []
            r_wrist_To_fingers = []
            offset = 7
            for i in range(n_lfingers):
                finger = lfingers_vals[offset:offset+7]
                finger_se3 = pin.XYZQUATToSE3(np.array(finger))
                wristMfingers = l_wrist_se3.inverse() * finger_se3
                l_wrist_To_fingers.append(wristMfingers.translation.tolist())
            for i in range(n_rfingers):
                finger = rfingers_vals[offset:offset+7]
                finger_se3 = pin.XYZQUATToSE3(np.array(finger))
                wristMfingers = r_wrist_se3.inverse() * finger_se3
                r_wrist_To_fingers.append(wristMfingers.translation.tolist())
                offset += 7
    
            # 3. Retargeting 실행 (qpos 결과 획득)
            lfingers_qdes = dex_retargeting.retarget_ref(
                dex_retargeting.left_retargeting, 
                np.array(l_wrist_To_fingers)
            )
            # rfingers_qdes = dex_retargeting.retarget_ref(
            #     dex_retargeting.right_retargeting, 
            #     np.array(r_wrist_To_fingers)
            # )

            # 4. 지정된 순서대로 Rviz에 Publish
            # left_retargeting_joint_names 순서로 나온 qpos를 
            # JointPublisher 내부의 target_joint_names 순서로 재정렬하여 발행
            lfingers_joint_pub.publish_joints(lfingers_qdes, dex_retargeting.left_retargeting_joint_names)
            # rfingers_joint_pub.publish_joints(rfingers_qdes, dex_retargeting.right_retargeting_joint_names)

            # 프레임 인덱스 업데이트
            lfingers_frame_idx = (lfingers_frame_idx + 1) % len(lfingers_data)
            rfingers_frame_idx = (rfingers_frame_idx + 1) % len(rfingers_data)

            # rclpy 스핀 및 루프 속도 조절 (예: 30Hz)
            rclpy.spin_once(lfingers_joint_pub, timeout_sec=0.033)
            rclpy.spin_once(rfingers_joint_pub, timeout_sec=0.033)

    except KeyboardInterrupt:
        pass
    finally:
        lfingers_joint_pub.destroy_node()
        rfingers_joint_pub.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()