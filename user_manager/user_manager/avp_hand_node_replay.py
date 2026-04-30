import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose, PoseArray
from sensor_msgs.msg import Image

# pinocchio
import pinocchio as pin
from pinocchio.utils import *

from avp_stream.streamer import VisionProStreamer

import cv2
from cv_bridge import CvBridge

bridge = CvBridge()

class AVPHandNode(Node):
    def __init__(self):
        super().__init__('avp_hand_node')
        self.get_logger().info('AVP Hand Node has been started.')

        self.default_fps = 30
        self.default_disparity = 30
        # self.default_resolution = "848x480" # same with the mjc renderer
        self.default_resolution = "640x480" # same with the mjc renderer
        # self.frame = np.ones((480,848,3))
        self.frame = np.ones((480,640,3))
        self.streamer = VisionProStreamer(ip="192.168.123.5")
        # Register stereo frame callback with custom disparity scale
        # self.streamer.register_frame_callback(self.create_rgb_visualizer)
        # self.streamer.configure_video(
        #     fps=self.default_fps,
        #     size=self.default_resolution,     # Side-by-side stereo resolution
        #     stereo=True,              # Enable stereo video mode
        # )
        self.streamer.start_webrtc(port=9999)
        # self.streamer.start_streaming()

        self.latest = None
        self.lfingers_json_path = '/home/home/mujoco_ws/src/user_manager/user_manager/src/data/test_lfingers_avp.json'
        self.lfingers_data_ = []

        # self.head_publisher = self.create_publisher(Pose, '/head', 10)
        self.lwrist_publisher = self.create_publisher(Pose, '/lwrist', 10)
        self.rwrist_publisher = self.create_publisher(Pose, '/rwrist', 10)
        self.lfingers_publisher = self.create_publisher(PoseArray, '/lfingers', 10)
        self.rfingers_publisher = self.create_publisher(PoseArray, '/rfingers', 10)

        self.timer = self.create_timer(0.01, self.timer_callback)

        self.pre_process = np.array([[0,1,0,0],[-1,0,0,0],[0,0,1,0],[0,0,0,1]])
        self.head_rot_diff = np.linalg.inv([[0,1,0],[-1,0,0],[0,0,1]])
        self.lwrist_rot_diff = np.linalg.inv([[1,0,0],[0,-1,0],[0,0,-1]])
        self.rwrist_rot_diff = np.linalg.inv([[-1,0,0],[0,-1,0],[0,0,1]])
        self.fingers_rot_diff = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]) @ np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])

    def timer_callback(self):
        self.latest = self.streamer.get_latest()
        if self.latest is None:
            self.get_logger().info('Waiting for pose data...')
            return
        # self.head_publish()
        self.lwrist_publish()
        self.rwrist_publish()
        self.lfingers_publish()
        self.rfingers_publish()

    # def head_publish(self):
    #     # post process
    #     head_data = self.latest.get("head")[0]
    #     head_data = self.pre_process @ head_data.copy()
    #     head_data[:3,:3] = head_data[:3,:3].copy() @ self.head_rot_diff

    #     head_msg = Pose()
    #     xyzquat = pin.SE3ToXYZQUAT(pin.SE3(head_data))
    #     head_msg.position.x = xyzquat[0]
    #     head_msg.position.y = xyzquat[1]
    #     head_msg.position.z = xyzquat[2]
    #     head_msg.orientation.x = xyzquat[3]
    #     head_msg.orientation.y = xyzquat[4]
    #     head_msg.orientation.z = xyzquat[5]
    #     head_msg.orientation.w = xyzquat[6]
    #     self.head_publisher.publish(head_msg)

    def lwrist_publish(self):
        # post process
        lwrist_data = self.latest.get("left_wrist")[0]
        lwrist_data = self.pre_process @ lwrist_data.copy()
        lwrist_data[:3,:3] = lwrist_data[:3,:3].copy() @ self.lwrist_rot_diff

        wrist_msg = Pose()
        xyzquat = pin.SE3ToXYZQUAT(pin.SE3(lwrist_data))
        wrist_msg.position.x = xyzquat[0]
        wrist_msg.position.y = xyzquat[1]
        wrist_msg.position.z = xyzquat[2]
        wrist_msg.orientation.x = xyzquat[3]
        wrist_msg.orientation.y = xyzquat[4]
        wrist_msg.orientation.z = xyzquat[5]
        wrist_msg.orientation.w = xyzquat[6]
        self.lwrist_publisher.publish(wrist_msg)

    def rwrist_publish(self):
        # post process
        rwrist_data = self.latest.get("right_wrist")[0]
        rwrist_data = self.pre_process @ rwrist_data.copy()
        rwrist_data[:3,:3] = rwrist_data[:3,:3].copy() @ self.rwrist_rot_diff

        wrist_msg = Pose()
        xyzquat = pin.SE3ToXYZQUAT(pin.SE3(rwrist_data))
        wrist_msg.position.x = xyzquat[0]
        wrist_msg.position.y = xyzquat[1]
        wrist_msg.position.z = xyzquat[2]
        wrist_msg.orientation.x = xyzquat[3]
        wrist_msg.orientation.y = xyzquat[4]
        wrist_msg.orientation.z = xyzquat[5]
        wrist_msg.orientation.w = xyzquat[6]
        self.rwrist_publisher.publish(wrist_msg)

    def lfingers_publish(self):
        # post process
        lfingers_data = self.latest.get("left_fingers")

        writs_rot = lfingers_data[0, :3, :3]
        writs_trans = lfingers_data[0, :3, 3]

        T = np.zeros((4,4))
        T[:3, :3] = writs_rot.T
        T[3,3] = 1

        x_rot = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])    
        y_rot = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])

        for i in range(len(lfingers_data)):
            lfingers_data[i,:,:] = T @ lfingers_data[i,:,:].copy()
            lfingers_data[i,:3,3] = lfingers_data[i,:3,3] @ x_rot @ y_rot

        self.lfingers_data_.append(lfingers_data)
        # lfingers_data[:,:3,3] = lfingers_data[:,:3,3].copy() @ self.fingers_rot_diff
        fingers_msg = PoseArray()
        for i in range(len(lfingers_data)):
            xyzquat = pin.SE3ToXYZQUAT(pin.SE3(lfingers_data[i]))
            pose = Pose()
            pose.position.x = xyzquat[0]
            pose.position.y = xyzquat[1]
            pose.position.z = xyzquat[2]
            pose.orientation.x = xyzquat[3]
            pose.orientation.y = xyzquat[4]
            pose.orientation.z = xyzquat[5]
            pose.orientation.w = xyzquat[6]
            fingers_msg.poses.append(pose)
        # self.get_logger().info(f'lfingers_data {lfingers_data} .')
        self.lfingers_publisher.publish(fingers_msg)

    def rfingers_publish(self):
        # post process
        rfingers_data = self.latest.get("right_fingers")
        # rfingers_data = self.pre_process @ rfingers_data.copy()
        # rfingers_data[:,:3,:3] = rfingers_data[:,:3,:3].copy() @ self.fingers_rot_diff
        fingers_msg = PoseArray()
        for i in range(len(rfingers_data)):
            xyzquat = pin.SE3ToXYZQUAT(pin.SE3(rfingers_data[i]))
            pose = Pose()
            pose.position.x = xyzquat[0]
            pose.position.y = xyzquat[1]
            pose.position.z = xyzquat[2]
            pose.orientation.x = xyzquat[3]
            pose.orientation.y = xyzquat[4]
            pose.orientation.z = xyzquat[5]
            pose.orientation.w = xyzquat[6]
            fingers_msg.poses.append(pose)
        self.rfingers_publisher.publish(fingers_msg)
        
    def save_data_to_json(self):
        import json
        lfingers_list = []
        for frame_data in self.lfingers_data_:
            frame_list = frame_data.tolist()
            lfingers_list.append({"data": frame_list})
        
        with open(self.lfingers_json_path, 'w') as f:
            json.dump(lfingers_list, f)
        self.get_logger().info(f'Saved lfingers data to {self.lfingers_json_path}')
    # def camera_callback(self, msg: Image):
    #     frame = bridge.imgmsg_to_cv2(msg)
    #     self.frame = np.fliplr(np.flipud(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)))

    # def create_rgb_visualizer(self, blank_frame):
    #     """Visualize streaming video"""
    #     np.copyto(blank_frame, self.frame)
    #     return blank_frame


def main():
    rclpy.init()
    node = AVPHandNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT) received.')
    finally:
        # 노드가 종료되기 직전에 데이터를 저장합니다.
        node.save_data_to_json()
        
        # 리소스 정리
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
