import rclpy
from rclpy.node import Node

from visualization_msgs.msg import Marker, MarkerArray

import numpy as np

# Pinocchio
import pinocchio as pin
from pinocchio.utils import *


class HandReplayNode(Node):
    def __init__(self):
        super().__init__('hand_replay_node')

        # ====== parameters ======
        self.file_path = '/data/avp_lwrist_fingers.txt'
        self.publish_hz = 30.0
        self.fixed_frame = 'base'

        # ====== publisher ======
        self.marker_pub = self.create_publisher(
            MarkerArray, '/hand_markers', 10
        )

        # ====== load data ======
        self.data = self.load_txt(self.file_path)
        if len(self.data) == 0:
            self.get_logger().error('TXT file is empty!')
            return

        self.frame_idx = 0
        self.n_fingers = (len(self.data[0]) - 7) // 7

        self.get_logger().info(
            f'Loaded {len(self.data)} frames, '
            f'{self.n_fingers} finger poses per frame'
        )

        # ====== timer ======
        self.timer = self.create_timer(
            1.0 / self.publish_hz, self.timer_callback
        )

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

    def msg_to_array(self, msg):
        x,y, z, qw, qx, qy, qz = msg.pose.position.x,msg.pose.position.y, msg.pose.position.z, msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z
        return np.array([x, y, z, qx, qy, qz, qw])
    


    def timer_callback(self):
        if len(self.data) == 0:
            return

        vals = self.data[self.frame_idx]
        self.frame_idx = (self.frame_idx + 1) % len(self.data)

        marker_array = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        marker_id = 0

        # ======================
        # Wrist marker
        # ======================
        wrist = vals[0:7]

        wrist_marker = Marker()
        wrist_marker.header.frame_id = self.fixed_frame
        wrist_marker.header.stamp = stamp
        wrist_marker.ns = 'wrist'
        wrist_marker.id = marker_id
        marker_id += 1

        wrist_marker.type = Marker.SPHERE
        wrist_marker.action = Marker.ADD


        wrist_marker.scale.x = 0.04
        wrist_marker.scale.y = 0.04
        wrist_marker.scale.z = 0.04

        wrist_marker.color.r = 1.0
        wrist_marker.color.g = 0.2
        wrist_marker.color.b = 0.2
        wrist_marker.color.a = 1.0

        # w_T_wrist
        # wrist_T_w-1 * w_T_wrist
        wrist_se3 = pin.XYZQUATToSE3(np.array(wrist))
        wrist_to_wrist = pin.SE3(1)
        
        wrist_marker.pose.position.x = wrist_to_wrist.translation[0]
        wrist_marker.pose.position.y = wrist_to_wrist.translation[1]
        wrist_marker.pose.position.z = wrist_to_wrist.translation[2]
        marker_array.markers.append(wrist_marker)

        # ======================
        # Finger markers
        # ======================
        offset = 7
        for i in range(self.n_fingers):
            # w_T_finger
            f = vals[offset:offset + 7]

            # wrist_T_finger
            wTf = wrist_se3.inverse() * pin.XYZQUATToSE3(np.array(f))
            wrist_to_f = wTf.translation

            offset += 7



            fm = Marker()
            fm.header.frame_id = self.fixed_frame
            fm.header.stamp = stamp
            fm.ns = 'fingers'
            fm.id = marker_id
            marker_id += 1

            fm.type = Marker.SPHERE
            fm.action = Marker.ADD

            fm.pose.position.x = wrist_to_f[0]
            fm.pose.position.y = wrist_to_f[1]
            fm.pose.position.z = wrist_to_f[2]

            fm.scale.x = 0.02
            fm.scale.y = 0.02
            fm.scale.z = 0.02

            fm.color.r = 0.2
            fm.color.g = 0.8
            fm.color.b = 1.0
            fm.color.a = 1.0

            marker_array.markers.append(fm)

        self.marker_pub.publish(marker_array)


def main():
    rclpy.init()
    node = HandReplayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
