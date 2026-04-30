import os
import argparse

import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState, Image

# MuJoCo
import mujoco
import mujoco.viewer

import numpy as np
from cv_bridge import CvBridge

from ament_index_python.packages import get_package_share_directory

bridge = CvBridge()

class MujocoSimulationIgrisCCubeNode(Node):
    def __init__(self, xml_file_name):
        super().__init__('mujoco_simulation_igris_c_cube_node')
        self.get_logger().info('Mujoco Simulation IgrisC Cube Node has been started.')
        self.get_logger().info(f'XML file: {xml_file_name}')

        # XML file
        descriptions_path = get_package_share_directory('igris_c_description')
        igrisc_xml_path = os.path.join(descriptions_path, xml_file_name)

        # Load MuJoCo model and data
        self.model = mujoco.MjModel.from_xml_path(igrisc_xml_path)
        if self.model is None:
            self.get_logger().error(f'Failed to load the model from {igrisc_xml_path}')
            return
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=480, width=640)

        self.publisher = self.create_publisher(JointState, '/mujoco/joint_states', 10)
        self.camera_publisher = self.create_publisher(Image, '/mujoco/camera', 10)
        self.subscriber = self.create_subscription(Float64MultiArray, '/mujoco/controller', self.joint_state_callback, 10)
        self.hand_subscriber = self.create_subscription(Float64MultiArray, '/mujoco/hand_controller', self.hand_joint_state_callback, 10)
        self.timer = self.create_timer(0.01, self.timer_callback)

        # Print initial joint information
        self.njnt = self.model.njnt # 15개
        print("njnt:", self.njnt)
        for i in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            qpos_adr = self.model.jnt_qposadr[i]
            qvel_adr = self.model.jnt_dofadr[i]
            print(f"[{i}] Joint: {joint_name}, qpos index: {qpos_adr}, qvel index: {qvel_adr}")

        self.ctrl = np.zeros(self.njnt-1)
        
        self.init = True
        self.ctrlFlag = False
    
    def joint_state_publish(self, qpos, qvel):
        joint_msg = JointState()
        left_qpos = qpos[7:14].tolist()
        left_qvel = qvel[6:13].tolist()
        right_qpos = qpos[25:32].tolist()
        right_qvel = qvel[24:31].tolist()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.position = left_qpos + right_qpos
        joint_msg.velocity = left_qvel + right_qvel
        self.publisher.publish(joint_msg)

    def camera_publish(self, camera_name='eef_camera'):
        # 해당 카메라의 뷰로 씬 업데이트
        self.renderer.update_scene(self.data, camera=camera_name)
        # RGB 이미지 렌더링 (numpy array 반환)
        pixels = self.renderer.render()
        # 여기서 ROS나 다른 통신 프로토콜로 frame을 publish 합니다.
        self.camera_publisher.publish(bridge.cv2_to_imgmsg(pixels))

    def joint_state_callback(self, msg: Float64MultiArray):
        for i in range(len(msg.data)):
            self.ctrl[i] = msg.data[i]
        self.ctrlFlag = True

    def hand_joint_state_callback(self, msg: Float64MultiArray):
        # print("Received hand joint states:", msg.name)
        self.ctrl[14:25] = msg.data[:11]
        self.ctrl[25:] = msg.data[11:]
        # self.ctrlFlag = True

    def timer_callback(self):
        # self.camera_publish()
        self.joint_state_publish(self.data.qpos, self.data.qvel)
        if self.init:
            print(self.data.qpos)
            # self.data.qpos = np.zeros((21))
            # self.data.qpos[10] = -np.pi/2
            # mujoco.mj_forward(self.model, self.data)
            self.init = False
        if self.ctrlFlag: # teleop_manager 를 통해서 joinstate 를 받을 예정
            self.data.qpos[7:14] = self.ctrl[:7]
            self.data.qpos[25:32] = self.ctrl[7:14]

            self.data.qpos[14:25] = self.ctrl[14:25]
            self.data.qpos[32:] = self.ctrl[25:]
            mujoco.mj_forward(self.model, self.data)

    def main(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while rclpy.ok() and viewer.is_running():
                rclpy.spin_once(self, timeout_sec=0.0)
                viewer.sync()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--xml', type=str, default='igris_c_v2.xml', help='MuJoCo XML file name')
    
    # ROS args랑 충돌 방지
    args, unknown = parser.parse_known_args()
    
    rclpy.init(args=unknown)

    node = MujocoSimulationIgrisCCubeNode(xml_file_name=args.xml)
    node.main()
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
