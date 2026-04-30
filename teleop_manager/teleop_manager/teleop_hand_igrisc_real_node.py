import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, PoseArray
from std_msgs.msg import Float32MultiArray
import pinocchio as pin
import numpy as np

# DexRetargeting 모듈 경로
from teleop_manager.src.dex_retargeting.DexRetargeting import DexRetargeting

class TeleopHandIgrisCRealNode(Node):
    def __init__(self):
        super().__init__('teleop_hand_igris_c_real_node')
        self.get_logger().info('Teleop Hand IgrisC Real Node has been started.')

        # 1. 리타겟팅 엔진 초기화
        self.dexretargeting = DexRetargeting('igris_c')

        # 2. 정규화 및 순서 정의
        self.desired_order = [
            "Thumb_Middle",
            "Index_Middle",
            "Middle_Middle",
            "Ring_Middle",
            "Little_Middle",
            "Thumb_Proximal"
        ]

        # URDF Joint Limits [Min, Max]
        self.limit_map = {
            "Thumb_Middle":   [0.0, 1.221730476],
            "Index_Middle":   [0.0, 1.570796327],
            "Middle_Middle":  [0.0, 1.570796327],
            "Ring_Middle":    [0.0, 1.570796327],
            "Little_Middle":  [0.0, 1.570796327],
            "Thumb_Proximal": [-1.727875959, 0.0]
        }

        # 3. 상태 변수
        self.l_wrist_pose = None
        self.r_wrist_pose = None
        self.l_fingers_poses = []
        self.r_fingers_poses = []

        # 4. 구독자 및 발행자 설정
        self.create_subscription(Pose, '/lwrist', self.lwrist_cb, 10)
        self.create_subscription(Pose, '/rwrist', self.rwrist_cb, 10)
        self.create_subscription(PoseArray, '/lfingers', self.lfingers_cb, 10)
        self.create_subscription(PoseArray, '/rfingers', self.rfingers_cb, 10)

        self.publisher = self.create_publisher(Float32MultiArray, '/real/igrisc/hand/command', 10)
        # self.publisher = self.create_publisher(Float32MultiArray, '/eun/command/hand', 10)
        self.timer = self.create_timer(0.01, self.timer_callback)
        
    def lwrist_cb(self, msg): self.l_wrist_pose = self.pose_to_se3(msg)
    def rwrist_cb(self, msg): self.r_wrist_pose = self.pose_to_se3(msg)
    def lfingers_cb(self, msg): self.l_fingers_poses = [self.pose_to_se3(p) for p in msg.poses]
    def rfingers_cb(self, msg): self.r_fingers_poses = [self.pose_to_se3(p) for p in msg.poses]

    def pose_to_se3(self, msg):
        return pin.XYZQUATToSE3([msg.position.x, msg.position.y, msg.position.z, 
                                 msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])

    def get_ordered_norm_values(self, qpos_raw, target_names, is_left=True):
        raw_data = {name: val for name, val in zip(target_names, qpos_raw)}
        ordered_values = []

        for key in self.desired_order:
            full_name = next((n for n in target_names if key in n), None)
            
            if full_name and full_name in raw_data:
                low, high = self.limit_map[key]
                val = raw_data[full_name]
                
                if key == "Thumb_Proximal":
                    # 절댓값이 커질수록 1, 작아질수록 0 (is_left 상관없이 공통 적용 가능)
                    # 공식: |val| / |MinLimit|
                    max_abs = abs(low) # 1.7278...
                    norm_val = abs(val) / max_abs
                else:
                    # 일반 조인트: (현재값 - Min) / (Max - Min)
                    norm_val = (val - low) / (high - low)
                
                ordered_values.append(float(np.clip(norm_val, 0.0, 1.0)))
            else:
                ordered_values.append(0.0)
        
        return ordered_values

    def timer_callback(self):
        if any(v is None for v in [self.l_wrist_pose, self.r_wrist_pose]) or \
           not self.l_fingers_poses or not self.r_fingers_poses:
            return

        l_ref = [(self.l_wrist_pose.inverse() * f).translation for f in self.l_fingers_poses]
        r_ref = [(self.r_wrist_pose.inverse() * f).translation for f in self.r_fingers_poses]

        l_q_raw = self.dexretargeting.retarget_ref(self.dexretargeting.left_retargeting, np.array(l_ref))[self.dexretargeting.left_retargeting_index]
        r_q_raw = self.dexretargeting.retarget_ref(self.dexretargeting.right_retargeting, np.array(r_ref))[self.dexretargeting.right_retargeting_index]

        final_left = self.get_ordered_norm_values(l_q_raw, self.dexretargeting.left_retargeting.optimizer.target_joint_names, is_left=True)
        final_right = self.get_ordered_norm_values(r_q_raw, self.dexretargeting.right_retargeting.optimizer.target_joint_names, is_left=False)

        msg = Float32MultiArray()
        msg.data = final_left + final_right
        self.publisher.publish(msg)
        
        # 확인용 로그 (필요시 주석 해제)
        self.get_logger().info(f"L_Hand: {np.round(final_left, 2)} | R_Hand: {np.round(final_right, 2)}")

def main():
    rclpy.init()
    node = TeleopHandIgrisCRealNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()