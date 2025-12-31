import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose

# pinocchio
import pinocchio as pin
from pinocchio.utils import *

from avp_stream.streamer import VisionProStreamer

class AVPNode(Node):
    def __init__(self):
        super().__init__('avp_node')
        self.get_logger().info('AVP Node has been started.')

        self.streamer = VisionProStreamer(ip="192.168.123.5")
        self.latest = None

        self.head_publisher = self.create_publisher(Pose, '/head', 10)
        self.lwrist_publisher = self.create_publisher(Pose, '/lwrist', 10)
        self.rwrist_publisher = self.create_publisher(Pose, '/rwrist', 10)
        # self.lfingers_publisher = self.create_publisher(Pose, '/lfingers', 10)
        # self.rfingers_publisher = self.create_publisher(Pose, '/rfingers', 10)

        self.timer = self.create_timer(0.01, self.timer_callback)
        
        self.pre_process = np.array([[0,1,0,0],[-1,0,0,0],[0,0,1,0],[0,0,0,1]])
        self.head_rot_diff = np.linalg.inv([[0,1,0],[-1,0,0],[0,0,1]])
        self.lwrist_rot_diff = np.linalg.inv([[1,0,0],[0,-1,0],[0,0,-1]])
        self.rwrist_rot_diff = np.linalg.inv([[-1,0,0],[0,-1,0],[0,0,1]])

    def timer_callback(self):
        self.latest = self.streamer.get_latest()
        if self.latest is None:
            self.get_logger().info('Waiting for pose data...')
            return
        self.head_publish()
        self.lwrist_publish()
        self.rwrist_publish()

    def head_publish(self):
        # post process
        head_data = self.latest.get("head")[0]
        head_data = self.pre_process @ head_data.copy()
        head_data[:3,:3] = head_data[:3,:3].copy() @ self.head_rot_diff

        head_msg = Pose()
        xyzquat = pin.SE3ToXYZQUAT(pin.SE3(head_data))
        head_msg.position.x = xyzquat[0]
        head_msg.position.y = xyzquat[1]
        head_msg.position.z = xyzquat[2]
        head_msg.orientation.x = xyzquat[3]
        head_msg.orientation.y = xyzquat[4]
        head_msg.orientation.z = xyzquat[5]
        head_msg.orientation.w = xyzquat[6]
        self.head_publisher.publish(head_msg)

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
            
def main():
    rclpy.init()
    node = AVPNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
