import rclpy
from rclpy.node import Node
import numpy as np

# 메시지 타입
from geometry_msgs.msg import Pose, PoseArray
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header

# Pinocchio
import pinocchio as pin
from pinocchio.utils import *

class VRHandMarkerVisualizer(Node):
    def __init__(self):
        super().__init__('vr_hand_marker_visualizer')
        self.lwrist_rot_diff_x = np.array([[1,0,0],[0,0,1],[0,-1,0]]).T
        # self.lwrist_rot_diff_x = np.array([[1,0,0],[0,0,-1],[0,1,0]]).T
        self.lwrist_rot_diff_y = np.array([[0,0,-1],[0,1,0],[1,0,0]]).T
        self.lwrist_rot_diff_z = np.array([[0,1,0],[-1,0,0],[0,0,1]]).T
        self.lfingers_rot_diff_se3 = pin.SE3(self.lwrist_rot_diff_x, np.zeros(3))
        # self.lwrist_rot_diff = np.array([[-1,0,0],[0,1,0],[0,0,-1]]).T
        # self.lfingers_rot_diff_se3 = pin.SE3(self.lwrist_rot_diff, np.zeros(3))
        # 1. 데이터 보관 변수
        self.wrist_pos = np.array([0.0, 0.0, 0.0])
        self.finger_positions = [] # List of np.array

        # 2. Subscriber 설정
        # /lwrist (손목 위치), /lfingers (손가락 끝점들 PoseArray)
        self.wrist_sub = self.create_subscription(Pose, '/lwrist', self.wrist_callback, 10)
        self.fingers_sub = self.create_subscription(PoseArray, '/lfingers', self.fingers_callback, 10)

        # 3. Publisher 설정
        # RViz에서 'MarkerArray' 디스플레이를 추가하고 이 토픽을 구독하세요.
        self.marker_pub = self.create_publisher(MarkerArray, '/vr_hand_markers', 10)

        # 4. 시각화 업데이트 타이머 (30Hz)
        self.timer = self.create_timer(0.033, self.publish_markers)
        
        self.get_logger().info("VR Hand Marker Visualizer Node Started")

    def wrist_callback(self, msg):
        self.wrist_pos = pin.XYZQUATToSE3([msg.position.x, msg.position.y, msg.position.z, msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z])

    def fingers_callback(self, msg):
        self.finger_positions = [pin.XYZQUATToSE3([p.position.x, p.position.y, p.position.z, p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z]) for p in msg.poses]

    def publish_markers(self):
        if len(self.finger_positions) == 0:
            return

        marker_array = MarkerArray()
        now = self.get_clock().now().to_msg()
        header = Header(frame_id="base", stamp=now)

        # --- 1. Wrist Marker (빨간색 큰 구) ---
        wrist_marker = Marker()
        wrist_marker.header = header
        wrist_marker.ns = "wrist"
        wrist_marker.id = 0
        wrist_marker.type = Marker.SPHERE
        wrist_marker.action = Marker.ADD

        wrist_to_wrist = pin.SE3(1)
        wrist_marker.pose.position.x, wrist_marker.pose.position.y, wrist_marker.pose.position.z = wrist_to_wrist.translation
        wrist_marker.scale.x = wrist_marker.scale.y = wrist_marker.scale.z = 0.04
        wrist_marker.color.r, wrist_marker.color.g, wrist_marker.color.b, wrist_marker.color.a = 1.0, 0.0, 0.0, 1.0
        marker_array.markers.append(wrist_marker)

        # --- 2. Finger Tips & Connections (녹색 구 및 선) ---
        for i, f_pos in enumerate(self.finger_positions):
            # 손가락 끝점 구체
            tip = Marker()
            tip.header = header
            tip.ns = "finger_tips"
            tip.id = i
            tip.type = Marker.SPHERE
            # wristTofingers = (self.wrist_pos).inverse() * f_pos
            wristTofingers = (self.wrist_pos*self.lfingers_rot_diff_se3).inverse() * f_pos
            tip.pose.position.x, tip.pose.position.y, tip.pose.position.z = wristTofingers.translation
            tip.scale.x = tip.scale.y = tip.scale.z = 0.02
            tip.color.r, tip.color.g, tip.color.b, tip.color.a = 0.0, 1.0, 0.0, 1.0
            marker_array.markers.append(tip)

            # 손목과 손가락 끝을 잇는 선 (Line Strip)
            # line = Marker()
            # line.header = header
            # line.ns = "connections"
            # line.id = i
            # line.type = Marker.LINE_STRIP
            # line.scale.x = 0.005 # 선 두께
            # line.color.r, line.color.g, line.color.b, line.color.a = 1.0, 1.0, 1.0, 0.6 # 흰색 반투명
            
            # # 선의 시작점(손목)과 끝점(손가락) 추가
            # p_wrist = Pose().position
            # p_wrist.x, p_wrist.y, p_wrist.z = wrist_to_wrist.translation
            # p_finger = Pose().position
            # p_finger.x, p_finger.y, p_finger.z = wristTofingers.translation
            
            # line.points.append(p_wrist)
            # line.points.append(p_finger)
            # marker_array.markers.append(line)

        self.marker_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = VRHandMarkerVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()