import os

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
from teleop_manager.src.dex_retargeting.DexRetargeting import DexRetargeting

bridge = CvBridge()

class MujocoSimulationNode(Node):
    def __init__(self):
        super().__init__('mujoco_simulation_upper_node')
        self.get_logger().info('Mujoco Simulation Upper Node has been started.')

        # XML file
        descriptions_path = get_package_share_directory('h1_2_description')
        h12_xml_path = os.path.join(descriptions_path, 'scene_upper.xml')

        # Load MuJoCo model and data
        self.model = mujoco.MjModel.from_xml_path(h12_xml_path)
        if self.model is None:
            self.get_logger().error(f'Failed to load the model from {h12_xml_path}')
            return
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=480, width=640)

        self.publisher = self.create_publisher(JointState, '/mujoco/joint_states', 10)
        self.camera_publisher = self.create_publisher(Image, '/mujoco/camera', 10)
        self.subscriber = self.create_subscription(Float64MultiArray, '/mujoco/controller', self.joint_state_callback, 10)
        self.timer = self.create_timer(0.01, self.timer_callback)

        # Print initial joint information
        self.dexretargeting = DexRetargeting("h1_2")
        self.left_retargeting_to_mjc = [] # [24, 22, 20, 18, 15, 14]
        self.right_retargeting_to_mjc = [] # [43, 41, 39, 37, 34, 33]

        mjc_joint_map = {}
        for i in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            qpos_adr = self.model.jnt_qposadr[i]
            qvel_adr = self.model.jnt_dofadr[i]
            print(f"[{i}] Joint: {joint_name}, qpos index: {qpos_adr}, qvel index: {qvel_adr}")
            mjc_joint_map[joint_name] = qpos_adr

        for target_joint_name in self.dexretargeting.left_retargeting.optimizer.target_joint_names:
            l_full_name = "L_" + target_joint_name
            r_full_name = "R_" + target_joint_name
            self.left_retargeting_to_mjc.append(mjc_joint_map[l_full_name])
            self.right_retargeting_to_mjc.append(mjc_joint_map[r_full_name])
        self.ctrl = np.zeros(26)
        
        self.init = True
        self.ctrlFlag = False
    
    # 14 개  
    def joint_state_publish(self, qpos, qvel):
        joint_msg = JointState()
        lwrist_qpos = qpos[7:14].tolist()   # 7
        # lfingers_qpos = qpos[14:26].tolist() # 12 
        lwrist_qvel = qvel[6:13].tolist()
        # lfingers_qvel = qvel[13:25].tolist()    

        rwrist_qpos = qpos[26:33].tolist()  # 7
        # rfingers_qpos = qpos[33:].tolist()# 12
        rwrist_qvel = qvel[25:32].tolist()
        # rfingers_qvel = qvel[32:].tolist()

        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.position = lwrist_qpos + rwrist_qpos
        joint_msg.velocity = lwrist_qvel + rwrist_qvel

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

    # 26 개
    def timer_callback(self):
        # self.camera_publish()
        self.joint_state_publish(self.data.qpos, self.data.qvel)
        if self.init:
            # print(len(self.data.qpos))
            self.init = False
        if self.ctrlFlag: # teleop_manager 를 통해서 joinstate 를 받을 예정
            # Left Wrist
            self.data.qpos[7:14] = self.ctrl[:7]
            # Right Wrist
            self.data.qpos[26:33] = self.ctrl[13:20]

            # Left Hand [24, 22, 20, 18, 15, 14]
            self.data.qpos[self.left_retargeting_to_mjc] = self.ctrl[7:13] # 6

            # Thumb
            self.data.qpos[16] = 1.334 * self.data.qpos[15] # intermediate
            self.data.qpos[17] = 0.667 * self.data.qpos[15] # distal
            
            # Others (Proximal -> Intermediate)
            self.data.qpos[25] = 1.06399 * self.data.qpos[24] # pinky
            self.data.qpos[23] = 1.06399 * self.data.qpos[22] # ring
            self.data.qpos[21] = 1.06399 * self.data.qpos[20] # middle
            self.data.qpos[19] = 1.06399 * self.data.qpos[18] # index
            
            # Right Hand [43, 41, 39, 37, 34, 33]
            self.data.qpos[self.right_retargeting_to_mjc] = self.ctrl[20:] # 6
            
            # Thumb: pitch에 종속
            self.data.qpos[35] = 1.334 * self.data.qpos[34] # R_thumb_intermediate_joint
            self.data.qpos[36] = 0.667 * self.data.qpos[34] # R_thumb_distal_joint
            
            # Fingers: proximal에 종속
            self.data.qpos[44] = 1.06399 * self.data.qpos[43] # R_pinky_intermediate
            self.data.qpos[42] = 1.06399 * self.data.qpos[41] # R_ring_intermediate
            self.data.qpos[40] = 1.06399 * self.data.qpos[39] # R_middle_intermediate
            self.data.qpos[38] = 1.06399 * self.data.qpos[37] # R_index_intermediate

            mujoco.mj_forward(self.model, self.data)
            
    def main(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while rclpy.ok() and viewer.is_running():
                rclpy.spin_once(self, timeout_sec=0.0)
                viewer.sync()
                # mujoco.mj_step(self.model, self.data)
                # self.streamer.update_sim() 

def main():
    rclpy.init()
    node = MujocoSimulationNode()
    node.main()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
