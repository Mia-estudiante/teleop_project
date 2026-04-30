import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose, PoseArray

# pinocchio
import pinocchio as pin
from pinocchio.utils import *

class AVPIgrisCNode(Node):
    def __init__(self):
        super().__init__('avp_igris_c_node')
        self.get_logger().info('AVP IgrisC Node has been started.')

        self.head_publisher = self.create_publisher(Pose, '/head', 10)
        self.lwrist_publisher = self.create_publisher(Pose, '/lwrist', 10)
        self.rwrist_publisher = self.create_publisher(Pose, '/rwrist', 10)
        self.lfingers_publisher = self.create_publisher(PoseArray, '/lfingers', 10)
        self.rfingers_publisher = self.create_publisher(PoseArray, '/rfingers', 10)

        self.avp_head_subscriber = self.create_subscription(Pose, '/avp/head', self.avp_head_callback, 10)
        self.avp_lwrist_subscriber = self.create_subscription(Pose, '/avp/lwrist', self.avp_lwrist_callback, 10)
        self.avp_rwrist_subscriber = self.create_subscription(Pose, '/avp/rwrist', self.avp_rwrist_callback, 10)
        self.avp_lfingers_subscriber = self.create_subscription(PoseArray, '/avp/lfingers', self.avp_lfingers_callback, 10)
        self.avp_rfingers_subscriber = self.create_subscription(PoseArray, '/avp/rfingers', self.avp_rfingers_callback, 10)

        self.timer = self.create_timer(0.01, self.timer_callback)

        self.head_goal = pin.SE3()
        self.l_goal = pin.SE3()
        self.r_goal = pin.SE3()
        self.lfingers_goal = []
        self.rfingers_goal = []

        self.avp_head_ctrlFlag = False
        self.avp_lwrist_ctrlFlag = False    
        self.avp_rwrist_ctrlFlag = False      
        self.avp_lfingers_ctrlFlag = False
        self.avp_rfingers_ctrlFlag = False       

        self.lwrist_rot_diff = np.array([[1,0,0],[0,0,1],[0,-1,0]]).T @ np.array([[0,0,-1],[0,1,0],[1,0,0]])
        self.rwrist_rot_diff = np.array([[1,0,0],[0,0,-1],[0,1,0]]).T @ np.array([[0,0,-1],[0,1,0],[1,0,0]])

    def timer_callback(self):
        if not(self.avp_head_ctrlFlag and self.avp_lwrist_ctrlFlag and self.avp_rwrist_ctrlFlag and self.avp_lfingers_ctrlFlag and self.avp_rfingers_ctrlFlag):
            print(f"{self.avp_head_ctrlFlag}, {self.avp_lwrist_ctrlFlag}, {self.avp_rwrist_ctrlFlag}, {self.avp_lfingers_ctrlFlag}, {self.avp_rfingers_ctrlFlag}")
            self.get_logger().info('Waiting for avp pose data...')
            return
        self.head_publish()
        self.lwrist_publish()
        self.rwrist_publish()
        self.lfingers_publish()
        self.rfingers_publish()

    def head_publish(self):
        head_data = self.head_goal
        # head_data = self.pre_process @ head_data.copy()
        # head_data[:3,:3] = head_data[:3,:3].copy() @ self.head_rot_diff

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
        lwrist_data = np.array(self.l_goal)
        # lwrist_data = self.pre_process @ lwrist_data.copy()
        # lwrist_data[:3,:3] = lwrist_data[:3,:3].copy() @ self.lwrist_rot_diff

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
        rwrist_data = np.array(self.r_goal)
        # rwrist_data = self.pre_process @ rwrist_data.copy()
        # rwrist_data[:3,:3] = rwrist_data[:3,:3].copy() @ self.rwrist_rot_diff

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
        lfingers_data = self.lfingers_goal # SE3 리스트

        fingers_msg = PoseArray()
        for i in range(len(lfingers_data)):
            finger_msg = Pose()
            lfingers_data[i].rotation = lfingers_data[i].rotation
            xyzquat = pin.SE3ToXYZQUAT(lfingers_data[i])
            finger_msg.position.x = xyzquat[0]
            finger_msg.position.y = xyzquat[1]
            finger_msg.position.z = xyzquat[2]
            finger_msg.orientation.x = xyzquat[3]
            finger_msg.orientation.y = xyzquat[4]
            finger_msg.orientation.z = xyzquat[5]
            finger_msg.orientation.w = xyzquat[6]
            fingers_msg.poses.append(finger_msg)
        self.lfingers_publisher.publish(fingers_msg)

    def rfingers_publish(self):
        rfingers_data = self.rfingers_goal

        fingers_msg = PoseArray()
        for i in range(len(rfingers_data)):
            finger_msg = Pose()
            rfingers_data[i].rotation = rfingers_data[i].rotation
            xyzquat = pin.SE3ToXYZQUAT(rfingers_data[i])
            finger_msg.position.x = xyzquat[0]
            finger_msg.position.y = xyzquat[1]
            finger_msg.position.z = xyzquat[2]
            finger_msg.orientation.x = xyzquat[3]
            finger_msg.orientation.y = xyzquat[4]
            finger_msg.orientation.z = xyzquat[5]
            finger_msg.orientation.w = xyzquat[6]
            fingers_msg.poses.append(finger_msg)
        self.rfingers_publisher.publish(fingers_msg)
            
    def avp_head_callback(self, msg: Pose):
        pos = np.array([msg.position.x, msg.position.y, msg.position.z, msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        self.head_goal = pin.XYZQUATToSE3(pos)
        self.avp_head_ctrlFlag = True

    def avp_lwrist_callback(self, msg: Pose):
        pos = np.array([msg.position.x, msg.position.y, msg.position.z, msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        self.l_goal = pin.XYZQUATToSE3(pos)
        self.avp_lwrist_ctrlFlag = True

    def avp_rwrist_callback(self, msg: Pose):
        pos = np.array([msg.position.x, msg.position.y, msg.position.z, msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        self.r_goal = pin.XYZQUATToSE3(pos)
        self.avp_rwrist_ctrlFlag = True

    def avp_lfingers_callback(self, msg: PoseArray):
        self.lfingers_goal = [
            pin.XYZQUATToSE3(np.array([
                p.position.x, p.position.y, p.position.z,
                p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w
            ]))
            for p in msg.poses
        ]
        self.avp_lfingers_ctrlFlag = True

    def avp_rfingers_callback(self, msg: PoseArray):
        self.rfingers_goal = [
            pin.XYZQUATToSE3(np.array([
                p.position.x, p.position.y, p.position.z,
                p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w
            ]))
            for p in msg.poses
        ]
        self.avp_rfingers_ctrlFlag = True
        
def main():
    rclpy.init()
    node = AVPIgrisCNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
