import os
import json

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose

# pinocchio
import pinocchio as pin
from pinocchio.utils import *

from ament_index_python.packages import get_package_share_directory

# --- 데이터 로드 함수 (가장 작은 프레임 수 확인용) ---
def load_data_from_json(filepath):
    with open(filepath, 'r') as f:
        json_data = json.load(f)

    all_frames_data = []
    for frame_obj in json_data:
        raw_data = frame_obj["data"]
        # 리스트가 한 번 더 감싸져 있는 경우(예: [[4x4]]) 처리
        data_array = np.array(raw_data)

        # (1, 4, 4) 형태로 들어오면 (4, 4)로 squeeze
        if data_array.ndim == 3 and data_array.shape[0] == 1:
            data_array = data_array.squeeze(axis=0)

        all_frames_data.append(data_array)
    return all_frames_data

class UserNode(Node):
    def __init__(self):
        super().__init__('user_node')
        self.get_logger().info('User Node has been started.')

        pkg = get_package_share_directory('user_manager')
        user_data_path = os.path.join(pkg, 'data')

        self.lwrist_path = os.path.join(user_data_path, 'test_lwrist.json')
        self.rwrist_path = os.path.join(user_data_path, 'test_rwrist.json')
        # self.lfingers_path = os.path.join(user_data_path, 'left_fingers.json')
        # self.rfingers_path = os.path.join(user_data_path, 'right_fingers.json')

        self.lwrist_publisher = self.create_publisher(Pose, '/lwrist', 10)
        self.rwrist_publisher = self.create_publisher(Pose, '/rwrist', 10)
        # self.lfingers_publisher = self.create_publisher(Pose, '/lfingers', 10)
        # self.rfingers_publisher = self.create_publisher(Pose, '/rfingers', 10)

        self.timer = self.create_timer(0.01, self.timer_callback)

        # Load data
        files = {
            "l_wrist": self.lwrist_path,
            "r_wrist": self.rwrist_path,
            # "l_fingers": self.lfingers_path,
            # "r_fingers": self.rfingers_path
        }
        
        data_frames = {}
        for key, path in files.items():
            data_frames[key] = load_data_from_json(path)

        self.lwrist_data_frames = data_frames['l_wrist']
        self.rwrist_data_frames = data_frames['r_wrist']

    def timer_callback(self):
        self.lwrist_publish(self.lwrist_data_frames)
        self.rwrist_publish(self.rwrist_data_frames)
        # self.lfingers_publish(self.lfingers_data_frames)
        # self.rfingers_publish(self.rfingers_data_frames)

    def lwrist_publish(self, frames):
        for frame in frames:
            wrist_msg = Pose()
            xyzquat = pin.SE3ToXYZQUAT(pin.SE3(np.array(frame)))
            wrist_msg.position.x = xyzquat[0]
            wrist_msg.position.y = xyzquat[1]
            wrist_msg.position.z = xyzquat[2]
            wrist_msg.orientation.x = xyzquat[3]
            wrist_msg.orientation.y = xyzquat[4]
            wrist_msg.orientation.z = xyzquat[5]
            wrist_msg.orientation.w = xyzquat[6]
            self.get_logger().info(f'lwrist_msg: {wrist_msg}')
            self.lwrist_publisher.publish(wrist_msg)

    def rwrist_publish(self, frames):
        for frame in frames:
            wrist_msg = Pose()
            xyzquat = pin.SE3ToXYZQUAT(pin.SE3(np.array(frame)))
            wrist_msg.position.x = xyzquat[0]
            wrist_msg.position.y = xyzquat[1]
            wrist_msg.position.z = xyzquat[2]
            wrist_msg.orientation.x = xyzquat[3]
            wrist_msg.orientation.y = xyzquat[4]
            wrist_msg.orientation.z = xyzquat[5]
            wrist_msg.orientation.w = xyzquat[6]
            self.get_logger().info(f'rwrist_msg: {wrist_msg}')
            self.rwrist_publisher.publish(wrist_msg)

def main():
    rclpy.init()
    node = UserNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
