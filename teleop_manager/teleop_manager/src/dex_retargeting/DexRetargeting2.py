import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

class IgrisFullJointBridge(Node):
    def __init__(self):
        super().__init__('igris_full_joint_bridge')
        self.publisher = self.create_publisher(JointState, 'joint_states', 10)
        
        # 1. RViz에서 제공된 전체 조인트 이름 리스트 (순서 고정)
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

        # 2. 파일 데이터 매핑 가이드 (파일 내 6개 값의 순서)
        # 텍스트 데이터 순서: pinky, ring, middle, index, thumb_p, thumb_y
        self.l_file_map = {
            'Left_9_Joint_Little_Middle': 0, 'Left_7_Joint_Ring_Middle': 1, 'Left_5_Joint_Middle_Middle': 2,
            'Left_3_Joint_Index_Middle': 3, 'Left_1_Joint_Thumb_Middle': 4, 'Left_0_Joint_Thumb_Proximal': 5
        }
        self.r_file_map = {
            'Right_9_Joint_Little_Middle': 0, 'Right_7_Joint_Ring_Middle': 1, 'Right_5_Joint_Middle_Middle': 2,
            'Right_3_Joint_Index_Middle': 3, 'Right_1_Joint_Thumb_Middle': 4, 'Right_0_Joint_Thumb_Proximal': 5
        }

        # 3. Mimic 매핑 (텍스트에 없는 끝마디 각도 복사)
        self.mimic_map = {
            'Left_10_Joint_Little_Distal': 'Left_9_Joint_Little_Middle',
            'Left_8_Joint_Ring_Distal': 'Left_7_Joint_Ring_Middle',
            'Left_6_Joint_Middle_Distal': 'Left_5_Joint_Middle_Middle',
            'Left_4_Joint_Index_Distal': 'Left_3_Joint_Index_Middle',
            'Left_2_Joint_Thumb_Distal': 'Left_1_Joint_Thumb_Middle',
            'Right_10_Joint_Little_Distal': 'Right_9_Joint_Little_Middle',
            'Right_8_Joint_Ring_Distal': 'Right_7_Joint_Ring_Middle',
            'Right_6_Joint_Middle_Distal': 'Right_5_Joint_Middle_Middle',
            'Right_4_Joint_Index_Distal': 'Right_3_Joint_Index_Middle',
            'Right_2_Joint_Thumb_Distal': 'Right_1_Joint_Thumb_Middle'
        }

        # 파일 로드 및 인덱스 초기화
        base_path = '/home/home/mujoco_ws/src/user_manager/user_manager/src/data/'
        self.l_data = self.load_data(os.path.join(base_path, 'retargeted_joints_left_igris.txt'))
        self.r_data = self.load_data(os.path.join(base_path, 'retargeted_joints_right_igris.txt'))
        self.idx = 0

        self.timer = self.create_timer(0.05, self.timer_callback)

    def load_data(self, path):
        if not os.path.exists(path):
            self.get_logger().error(f"File not found: {path}")
            return []
        with open(path, 'r') as f:
            return [line.strip() for line in f.readlines() if line.strip() and not line.startswith('#')]

    def timer_callback(self):
        if not self.l_data or not self.r_data: return

        l_vals = list(map(float, self.l_data[self.idx % len(self.l_data)].split()))
        r_vals = list(map(float, self.r_data[self.idx % len(self.r_data)].split()))

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.all_joint_names
        # 전체 53개 조인트에 대해 기본값 0.0 할당
        positions = [0.0] * len(self.all_joint_names)

        # 왼손 메인 조인트 값 업데이트
        for j_name, f_idx in self.l_file_map.items():
            positions[self.all_joint_names.index(j_name)] = l_vals[f_idx]

        # 오른손 메인 조인트 값 업데이트
        for j_name, f_idx in self.r_file_map.items():
            positions[self.all_joint_names.index(j_name)] = r_vals[f_idx]

        # Mimic 조인트 업데이트 (Distal 마디 시각화)
        for child, parent in self.mimic_map.items():
            p_idx = self.all_joint_names.index(parent)
            positions[self.all_joint_names.index(child)] = positions[p_idx]

        msg.position = positions
        self.publisher.publish(msg)
        self.idx += 1

def main():
    rclpy.init()
    node = IgrisFullJointBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()