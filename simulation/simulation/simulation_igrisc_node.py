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

class MujocoSimulationIgrisCNode(Node):
    def __init__(self, xml_file_name='igris_c_v2.xml'):
        super().__init__('mujoco_simulation_igris_c_node')
        self.get_logger().info('Mujoco Simulation IgrisC Node has been started.')
        self.get_logger().info(f'XML file: {xml_file_name}')
        
        descriptions_path = get_package_share_directory('igris_c_description')
        '''
        # XML file
        igrisc_xml_path = os.path.join(descriptions_path, xml_file_name)

        # Load MuJoCo model and data
        self.model = mujoco.MjModel.from_xml_path(igrisc_urdf_path)
        '''
        igrisc_urdf_path = os.path.join(descriptions_path, 'urdf/igris_c_v2_hand.urdf')

        pkg_path = get_package_share_directory('igris_c_description') + '/'

        print(f"pkg path: {pkg_path}")
        # URDF 읽어서 package:// 치환
        with open(igrisc_urdf_path, 'r') as f:
            urdf_str = f.read()
        urdf_str = urdf_str.replace('package://igris_c_description/', pkg_path)

        # 2. visual mesh 보존
        urdf_str = urdf_str.replace(
            '<robot name="IGRIS_C">',
            '<robot name="IGRIS_C">\n'
            '  <mujoco>\n'
            '    <compiler discardvisual="false"/>\n'
            '  </mujoco>'
        )

        # 3. 색상 밝게 (모든 rgba 값 일괄 치환)
        urdf_str = re.sub(
            r'rgba="0\.0706 0\.0706 0\.0706 0\.7"',
            'rgba="0.75 0.75 0.78 1.0"',
            urdf_str
        )

        self.model = mujoco.MjModel.from_xml_string(urdf_str)

        non_mesh = self.model.geom_type != mujoco.mjtGeom.mjGEOM_MESH
        self.model.geom_rgba[non_mesh, 3] = 0.0  # 알파 = 0


        if self.model is None:
            self.get_logger().error(f'Failed to load the model from {igrisc_urdf_path}')
            return
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

        # Print initial joint information
        self.njnt = self.model.njnt # 15개
        print("njnt:", self.njnt)
        for i in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            qpos_adr = self.model.jnt_qposadr[i]
            qvel_adr = self.model.jnt_dofadr[i]
            print(f"[{i}] Joint: {joint_name}, qpos index: {qpos_adr}, qvel index: {qvel_adr}")

        if xml_file_name == 'igris_c_v2.xml':
            self.ctrl = np.zeros(self.njnt-1)
        elif xml_file_name == 'igris_c_v2_cube.xml':
            self.ctrl = np.zeros(self.njnt-2)
            self.target_ctrl = np.zeros(self.njnt-2)   # 가장 최근 VR에서 온 명령
            self.smooth_ctrl = np.zeros(self.njnt-2)   # MuJoCo에 실제 들어가는 값
        self.init = True
        self.ctrlFlag = False

        self.kp = 200.0
        self.kd = 2.0
        # self.tau_max = 20.0
        self.alpha = 0.05

        # 코드의 PD 게인
        # arm_kp = np.array([400, 400, 400, 400,  400, 400,  400])
        arm_kp = np.array([300, 300, 200, 250,  40,  40,  40])
        # arm_kd = np.array([ 15,  15,  10,  12,   1,   1,   1])  # damping이 이미 들어가 있으니 작게
        arm_kd = np.array([ 12,  12,   8,  10,   2,   2,   2])

        self.kp = np.concatenate([arm_kp, arm_kp])  # 14
        self.kd = np.concatenate([arm_kd, arm_kd])
        # 모터 스펙을 코드에 명시 (xml ctrlrange와 일치시킴)
        arm_tau_max = np.array([60, 60, 60, 60, 8, 8, 8])  # 7 joints per arm
        self.tau_max = np.concatenate([arm_tau_max, arm_tau_max])  # 14
    
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
    '''
    '''
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

            # q_err = self.ctrl[:] - self.data.qpos[7:21]
            # qd    = self.data.qvel[6:20]
            # tau   = self.kp * q_err - self.kd * qd
            # self.data.ctrl[:] = tau
            # self.data.ctrl[:] = np.clip(tau, -self.tau_max, self.tau_max)
            # print("tau:", tau)
            # print("clip tau:", self.data.ctrl[:])
            # mujoco.mj_step(self.model, self.data)

            # tau = self.kp * (self.ctrl[:] - self.data.qpos[7:21]) - self.kd * self.data.qvel[6:20]
            # self.data.ctrl[:] = np.clip(tau, -self.tau_max, self.tau_max)

            # self.data.qpos[7:14] = self.ctrl[:7]
            # self.data.qpos[14:21] = self.ctrl[7:]
            # mujoco.mj_forward(self.model, self.data)
    
    # def timer_callback(self):
    #     self.joint_state_publish(self.data.qpos, self.data.qvel)
    #     print(self.data.qpos)
        
    #     if self.ctrlFlag:
    #         tau = self.kp * (self.ctrl[:] - self.data.qpos[7:21]) - self.kd * self.data.qvel[6:20]
    #         self.data.ctrl[:] = np.clip(tau, -self.tau_max, self.tau_max)
        
    #     # 100Hz timer × 20 substeps × 0.0005s timestep = 1.0 (real time)
    #     n_sub = max(1, int(round(0.01 / self.model.opt.timestep)))
    #     for _ in range(n_sub):
    #         # PD를 매 substep마다 갱신하면 더 안정적
    #         if self.ctrlFlag:
    #             tau = self.kp * (self.ctrl[:] - self.data.qpos[7:21]) - self.kd * self.data.qvel[6:20]
    #             self.data.ctrl[:] = np.clip(tau, -self.tau_max, self.tau_max)
    #         mujoco.mj_step(self.model, self.data)

    # def timer_callback(self):
    #     self.joint_state_publish(self.data.qpos, self.data.qvel)
    #     print(self.data.qpos)
        
    #     # 100Hz timer × 20 substeps × 0.0005s timestep = 1.0 (real time)
    #     n_sub = max(1, int(round(0.01 / self.model.opt.timestep)))
    #     for _ in range(n_sub):
    #         self.data.ctrl[:] = self.ctrl[:]

    #         mujoco.mj_step(self.model, self.data)

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

    node = MujocoSimulationIgrisCNode(xml_file_name=args.xml)
    node.main()
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
