import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import matplotlib.pyplot as plt
import numpy as np

class CameraViewer(Node):
    def __init__(self):
        super().__init__('camera_viewer_node')
        self.subscription = self.create_subscription(
            Image,
            # '/mujoco/camera/front_view',
            '/mujoco/camera/side_view',
            # '/mujoco/camera/top_view',
            self.listener_callback,
            10)
        self.bridge = CvBridge()
        
        # Matplotlib 설정
        plt.ion()  # 인터랙티브 모드 활성화
        self.fig, self.ax = plt.subplots()
        self.im = None
        self.get_logger().info('Camera Viewer Node has been started.')

    def listener_callback(self, msg):
        try:
            # 1. 'passthrough'를 사용하여 원본 데이터(8UC3)를 그대로 가져옵니다.
            # 이렇게 하면 'encoding' 에러를 피할 수 있습니다.
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

            # 2. MuJoCo에서 보낸 데이터가 BGR 순서라면 RGB로 바꿔줍니다 (Matplotlib용)
            # 만약 색상이 이상하게 나오면(파란색과 빨간색이 바뀜) 아래 주석을 해제하세요.
            # import cv2
            # cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

            if self.im is None:
                self.im = self.ax.imshow(cv_image)
                plt.axis('off')
                plt.tight_layout()
            else:
                self.im.set_data(cv_image)
            
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            
        except Exception as e:
            self.get_logger().error(f'Could not convert image: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = CameraViewer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        plt.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()