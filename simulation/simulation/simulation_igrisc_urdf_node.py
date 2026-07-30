import os
from pyexpat import model
import re
import sys, argparse

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState, Image

# MuJoCo
import mujoco
import mujoco.viewer

import numpy as np
from cv_bridge import CvBridge

from ament_index_python.packages import get_package_share_directory

bridge = CvBridge()

class MujocoSimulationIgrisCURDFNode(Node):
    def __init__(self):
        super().__init__('mujoco_simulation_igris_c_urdf_node')
        self.get_logger().info('Mujoco Simulation IgrisC URDF Node has been started.')
        
        descriptions_path = get_package_share_directory('igris_c_description')
        igrisc_xml_path = os.path.join(descriptions_path, 'igris_c_v2_converted.xml')

        self.model = mujoco.MjModel.from_xml_path(igrisc_xml_path)
        if self.model is None:
            self.get_logger().error(f'Failed to load the model from {igrisc_xml_path}')
            return
        
        non_mesh = self.model.geom_type != mujoco.mjtGeom.mjGEOM_MESH
        self.model.geom_rgba[non_mesh, 3] = 0.0  # 알파 = 0

        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "initial")
        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)

        # simulation_igrisc_node.py
        ctrl_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,  # ← 변경
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.publisher = self.create_publisher(JointState, '/mujoco/joint_states', 10)
        self.camera_publisher = self.create_publisher(Image, '/mujoco/camera', 10)
        self.subscriber = self.create_subscription(Float64MultiArray, '/mujoco/controller', self.joint_state_callback, ctrl_qos)
        self.timer = self.create_timer(0.01, self.timer_callback)

        print(f"actuator 개수 (nu): {self.model.nu}")  # 14가 나와야 정상
        # 이름들도 확인
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            print(f"  [{i}] {name}")

        # Print initial joint information
        self.njnt = self.model.njnt # 15개
        print("njnt:", self.njnt)
        for i in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            qpos_adr = self.model.jnt_qposadr[i]
            qvel_adr = self.model.jnt_dofadr[i]
            print(f"[{i}] Joint: {joint_name}, qpos index: {qpos_adr}, qvel index: {qvel_adr}")

        self.init = True
        self.ctrlFlag = False
    
    def joint_state_publish(self, qpos, qvel):
        joint_msg = JointState()
        left_qpos = qpos[7:14].tolist()
        left_qvel = qvel[6:13].tolist()
        right_qpos = qpos[14:21].tolist()
        right_qvel = qvel[13:20].tolist()
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
            self.target_ctrl[i] = msg.data[i]
        self.ctrlFlag = True
        print("JOINT STATE CALL:", self.ctrl)

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
            print("self.data.ctrl:", self.data.ctrl[:])
            print("self.ctrl:", self.ctrl)
            # self.data.qpos[7:14] = self.ctrl[:7]
            # self.data.qpos[14:21] = self.ctrl[7:]
            # self.data.ctrl[:] = self.ctrl[:]

            self.smooth_ctrl = (1 - self.alpha) * self.smooth_ctrl + self.alpha * self.target_ctrl
            self.data.ctrl[:] = self.smooth_ctrl
            mujoco.mj_step(self.model, self.data)

    def main(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while rclpy.ok() and viewer.is_running():
                rclpy.spin_once(self, timeout_sec=0.0)
                viewer.sync()

def main():
    rclpy.init()
    node = MujocoSimulationIgrisCURDFNode()
    node.main()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
