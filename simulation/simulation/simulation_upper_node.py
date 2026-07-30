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

'''
ctrl: 26개 - arm: 7개*2 + fingers: 6개*2
qpos: 45개 - qpos index: 7~44, 실질적 사용 joint 38개
njnt: 39개
'''
class MujocoSimulationNode(Node):
    def __init__(self, robot_name):
        super().__init__('mujoco_simulation_node')
        self.get_logger().info('Mujoco Simulation Node has been started.')
        
        self.robot_name = robot_name
        if robot_name=='h1_2':
            descriptions_path = get_package_share_directory('h1_2_description')
            robot_xml_path = os.path.join(descriptions_path, 'scene_hand.xml')
        elif robot_name=='igris_c':
            descriptions_path = get_package_share_directory('igris_c_description')
            robot_xml_path = os.path.join(descriptions_path, 'scene.xml')
            print("igris")

        # Load MuJoCo model and data
        self.model = mujoco.MjModel.from_xml_path(robot_xml_path)
        if self.model is None:
            self.get_logger().error(f'Failed to load the model from {robot_xml_path}')
            return
        self.data = mujoco.MjData(self.model)
        
        self.publisher = self.create_publisher(JointState, '/mujoco/joint_states', 10)
        self.subscriber = self.create_subscription(Float64MultiArray, '/mujoco/controller', self.joint_state_callback, 10)
        self.timer = self.create_timer(0.01, self.timer_callback)

        # self.njnt = self.model.njnt # 39개
        if robot_name == 'h1_2':
            from teleop_manager.src.dex_retargeting.DexRetargeting import DexRetargeting
            self.dexretargeting = DexRetargeting(robot_name)
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
        elif robot_name=='igris_c':
            for i in range(self.model.njnt):
                joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                qpos_adr = self.model.jnt_qposadr[i]
                qvel_adr = self.model.jnt_dofadr[i]
                print(f"[{i}] Joint: {joint_name}, qpos index: {qpos_adr}, qvel index: {qvel_adr}")

            self.ctrl = np.zeros(14)
        
        self.init = True
        self.ctrlFlag = False

    def joint_state_publish(self, qpos, qvel):
        joint_msg = JointState()
        if self.robot_name == 'h1_2':
            left_qpos = qpos[7:14].tolist()
            left_qvel = qvel[6:13].tolist()
            right_qpos = qpos[26:33].tolist()
            right_qvel = qvel[25:32].tolist()
        elif self.robot_name == 'igris_c':
            left_qpos = qpos[7:14].tolist()
            left_qvel = qvel[6:13].tolist()
            right_qpos = qpos[14:].tolist()
            right_qvel = qvel[13:].tolist()

        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.position = left_qpos + right_qpos
        joint_msg.velocity = left_qvel + right_qvel
        self.publisher.publish(joint_msg)

    def joint_state_callback(self, msg: Float64MultiArray):
        for i in range(len(msg.data)):
            # self.get_logger().info(f'Controller data[{i}]: {msg.data[i]}')
            self.ctrl[i] = msg.data[i]
        self.ctrlFlag = True
    
    # 26 개
    def timer_callback(self):
        self.joint_state_publish(self.data.qpos, self.data.qvel)
        if self.init:
            self.init = False
        if self.ctrlFlag: # teleop_manager 를 통해서 joinstate 를 받을 예정
            if self.robot_name == 'h1_2':
                self.data.qpos[7:14] = self.ctrl[:7]
                self.data.qpos[26:33] = self.ctrl[13:20]
                self.data.qpos[self.left_retargeting_to_mjc] = self.ctrl[7:13] # 6
                self.data.qpos[self.right_retargeting_to_mjc] = self.ctrl[20:] # 6

                # Thumb
                self.data.qpos[16] = 1.334 * self.data.qpos[15] # intermediate
                self.data.qpos[17] = 0.667 * self.data.qpos[15] # distal
                
                # Others (Proximal -> Intermediate)
                self.data.qpos[25] = 1.06399 * self.data.qpos[24] # pinky
                self.data.qpos[23] = 1.06399 * self.data.qpos[22] # ring
                self.data.qpos[21] = 1.06399 * self.data.qpos[20] # middle
                self.data.qpos[19] = 1.06399 * self.data.qpos[18] # index
                
                # Thumb: pitch에 종속 (multiplier 1.334, 0.667)
                self.data.qpos[35] = 1.334 * self.data.qpos[34] # R_thumb_intermediate_joint
                self.data.qpos[36] = 0.667 * self.data.qpos[34] # R_thumb_distal_joint
                
                # Fingers: proximal에 종속 (multiplier 1.06399, offset -0.04545)
                # mimic 공식: q_mimic = (q_src * multiplier) + offset
                self.data.qpos[44] = 1.06399 * self.data.qpos[43] # R_pinky_intermediate
                self.data.qpos[42] = 1.06399 * self.data.qpos[41] # R_ring_intermediate
                self.data.qpos[40] = 1.06399 * self.data.qpos[39] # R_middle_intermediate
                self.data.qpos[38] = 1.06399 * self.data.qpos[37] # R_index_intermediate
            elif self.robot_name == 'igris_c':
                self.data.qpos[7:] = self.ctrl[:]
            mujoco.mj_forward(self.model, self.data)

    def main(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while rclpy.ok() and viewer.is_running():
                rclpy.spin_once(self, timeout_sec=0.0)
                viewer.sync()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robot", choices=['h1_2','igris_c'], default='h1_2'
    )
    args, ros_args = parser.parse_known_args()
    rclpy.init(args=ros_args)
    node = MujocoSimulationNode(args.robot)
    node.main()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
