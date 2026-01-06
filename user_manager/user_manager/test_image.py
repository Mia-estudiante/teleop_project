import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image

# pinocchio
import pinocchio as pin
from pinocchio.utils import *

from avp_stream.streamer import VisionProStreamer

import cv2
from cv_bridge import CvBridge

bridge = CvBridge()


def add_text(frame):
    return cv2.putText(frame, "Hello VisionPro!", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0,255,0), 2)


class TestImageNode(Node):
    def __init__(self):
        super().__init__('test_image_node')
        self.get_logger().info('AVP Node has been started.')

        self.default_fps = 30
        self.default_disparity = 30
        self.default_resolution = "640x480"
        self.frame = np.ones((480,640,3))
        self.streamer = VisionProStreamer(ip="192.168.123.5")
        # Register stereo frame callback with custom disparity scale
        self.streamer.register_frame_callback(self.create_rgb_visualizer)
        '''
        # self.streamer.register_frame_callback(create_rgb_visualizer(self.streamer, disparity_scale=self.default_disparity))
        
        # Configure video streaming with stereo enabled
        # print(f"Starting stereo video stream at {args.default_resolution}, {args.default_fps} fps...")
        '''
        self.streamer.configure_video(
            fps=self.default_fps,
            size=self.default_resolution,     # Side-by-side stereo resolution
            stereo=True,              # Enable stereo video mode
        )
        self.streamer.start_webrtc(port=9999)
        # self.streamer.start_streaming()
        
        self.latest = None

        self.camera_subscriber = self.create_subscription(Image, '/mujoco/camera', self.camera_callback, 10)

        self.timer = self.create_timer(0.01, self.timer_callback)
    
    def camera_callback(self, msg: Image):
        frame = bridge.imgmsg_to_cv2(msg)
        self.frame = np.fliplr(np.flipud(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)))

        # self.image = bridge.imgmsg_to_cv2(msg)
        # print(type(self.image))
        # print(self.image.shape)
        # def stream_rgb(image):
        #     return np.fliplr(np.flipud(cv2.cvtColor(image, cv2.COLOR_RGB2BGR)))
        # return stream_rgb
        # cv2.imshow('image', np.fliplr(np.flipud(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))))
        # cv2.waitKey(2)  

    def create_rgb_visualizer(self, blank_frame):
        """Visualize streaming video"""
        np.copyto(blank_frame, self.frame)

        return blank_frame


    def timer_callback(self):
        pass
        # self.latest = self.streamer.get_latest()
        # if self.latest is None:
        #     self.get_logger().info('Waiting for pose data...')
        #     return

      

def main():
    rclpy.init()
    node = TestImageNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
