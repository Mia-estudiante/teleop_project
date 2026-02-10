import os
import numpy as np
import mujoco
import mujoco.viewer
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from ament_index_python.packages import get_package_share_directory

class MujocoSimulationHandNode(Node):
    def __init__(self):
        super().__init__('mujoco_simulation_hand_node')
        
        # 1. 모델 로드
        descriptions_path = get_package_share_directory('h1_2_description')
        h12_xml_path = os.path.join(descriptions_path, 'scene_.xml')
        self.model = mujoco.MjModel.from_xml_path(h12_xml_path)
        self.data = mujoco.MjData(self.model)

        # 2. TXT 데이터 로드 (경로를 본인 환경에 맞게 수정하세요)
        self.txt_file_path = '/data/retargeted_joints_left.txt'
        self.recorded_data = self.load_txt(self.txt_file_path)
        self.frame_idx = 0
        self.total_frames = len(self.recorded_data)
        
        self.publisher = self.create_publisher(JointState, '/mujoco/joint_states', 10)
        self.timer = self.create_timer(0.01, self.timer_callback) # 100Hz

    def load_txt(self, path):
        data = []
        try:
            with open(path, 'r') as f:
                for line in f:
                    if line.strip().startswith('#') or not line.strip():
                        continue
                    data.append(list(map(float, line.split())))
            self.get_logger().info(f"Successfully loaded {len(data)} frames from TXT.")
        except Exception as e:
            self.get_logger().error(f"Failed to load TXT: {e}")
        return data

    def timer_callback(self):
        if not self.recorded_data:
            return

        # 1. TXT에서 현재 프레임 데이터 가져오기
        # 순서: [0:pinky, 1:ring, 2:middle, 3:index, 4:pitch, 5:yaw]
        current_q = self.recorded_data[self.frame_idx]

        # 2. MuJoCo qpos에 직접 매핑 (제공해주신 인덱스 기준)
        # Proximal Joints
        self.data.qpos[24] = current_q[0]  # L_pinky_proximal_joint
        self.data.qpos[22] = current_q[1]  # L_ring_proximal_joint
        self.data.qpos[20] = current_q[2]  # L_middle_proximal_joint
        self.data.qpos[18] = current_q[3]  # L_index_proximal_joint
        self.data.qpos[15] = current_q[4]  # L_thumb_proximal_pitch_joint
        self.data.qpos[14] = current_q[5]  # L_thumb_proximal_yaw_joint

        # self.data.qpos[43] = current_q[0]  # R_pinky_proximal_joint (Index 43)
        # self.data.qpos[41] = current_q[1]  # R_ring_proximal_joint (Index 41)
        # self.data.qpos[39] = current_q[2]  # R_middle_proximal_joint (Index 39)
        # self.data.qpos[37] = current_q[3]  # R_index_proximal_joint (Index 37)
        # self.data.qpos[34] = current_q[4]  # R_thumb_proximal_pitch_joint (Index 34)
        # self.data.qpos[33] = current_q[5]  # R_thumb_proximal_yaw_joint (Index 33)

        # 3. Mimic Joints 계산 (종속 관절)
        # # Thumb
        self.data.qpos[16] = 1.334 * self.data.qpos[15] # intermediate
        self.data.qpos[17] = 0.667 * self.data.qpos[15] # distal
        
        # # Others (Proximal -> Intermediate)
        self.data.qpos[25] = 1.06399 * self.data.qpos[24] # pinky
        self.data.qpos[23] = 1.06399 * self.data.qpos[22] # ring
        self.data.qpos[21] = 1.06399 * self.data.qpos[20] # middle
        self.data.qpos[19] = 1.06399 * self.data.qpos[18] # index

        # Thumb: pitch에 종속 (multiplier 1.334, 0.667)
        # self.data.qpos[35] = 1.334 * self.data.qpos[34] # R_thumb_intermediate_joint
        # self.data.qpos[36] = 0.667 * self.data.qpos[34] # R_thumb_distal_joint
        
        # Fingers: proximal에 종속 (multiplier 1.06399, offset -0.04545)
        # mimic 공식: q_mimic = (q_src * multiplier) + offset
        # self.data.qpos[44] = (1.06399 * self.data.qpos[43])# R_pinky_intermediate
        # self.data.qpos[42] = (1.06399 * self.data.qpos[41])# R_ring_intermediate
        # self.data.qpos[40] = (1.06399 * self.data.qpos[39])# R_middle_intermediate
        # self.data.qpos[38] = (1.06399 * self.data.qpos[37])# R_index_intermediate


        # 4. 시뮬레이션 전향 계산 및 프레임 업데이트
        mujoco.mj_forward(self.model, self.data)
        self.frame_idx = (self.frame_idx + 1) % self.total_frames

        # 5. 상태 발행 (필요 시)
        self.publish_current_state()

    def publish_current_state(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        # 전체 qpos 중 손 부분만 잘라서 보낼 수도 있습니다.
        msg.position = self.data.qpos.tolist() 
        self.publisher.publish(msg)

    def main(self):
        # 뷰어 실행
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while rclpy.ok() and viewer.is_running():
                rclpy.spin_once(self, timeout_sec=0.0)
                viewer.sync()

def main():
    rclpy.init()
    node = MujocoSimulationHandNode()
    node.main()
    rclpy.shutdown()

if __name__ == '__main__':
    main()