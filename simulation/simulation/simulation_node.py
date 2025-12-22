import os

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState

# MuJoCo
import mujoco
import mujoco.viewer

import numpy as np

from ament_index_python.packages import get_package_share_directory

class MujocoSimulationNode(Node):
    def __init__(self):
        super().__init__('mujoco_simulation_node')
        self.get_logger().info('Mujoco Simulation Node has been started.')

        # XML file
        descriptions_path = get_package_share_directory('h1_2_description')
        h12_xml_path = os.path.join(descriptions_path, 'scene.xml')

        # Load MuJoCo model and data
        self.model = mujoco.MjModel.from_xml_path(h12_xml_path)
        if self.model is None:
            self.get_logger().error(f'Failed to load the model from {h12_xml_path}')
            return
        self.data = mujoco.MjData(self.model)
        print(f"data qpos: {self.data.qpos}")

        self.publisher = self.create_publisher(JointState, '/mujoco/joint_states', 10)
        self.subscriber = self.create_subscription(JointState, '/mujoco/controller', self.joint_state_callback, 10)
        self.timer = self.create_timer(0.01, self.timer_callback)

        # Print initial joint information
        self.njnt = self.model.njnt # 15개
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
        right_qpos = qpos[14:21].tolist()
        right_qvel = qvel[13:20].tolist()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.position = left_qpos + right_qpos
        joint_msg.velocity = left_qvel + right_qvel
        self.publisher.publish(joint_msg)
        
    def joint_state_callback(self, msg):
        for i in range(len(msg.position)):
            self.ctrl[i] = msg.position[i]
        self.ctrlFlag = True
        self.get_logger().info(f'Received joint state: {msg}')

    '''
    def joint_state_callback_test(self):
        self.ctrlFlag = True
        self.ctrl[:7] = np.array([0,0,1.03, 0,0,0,1])
        # self.ctrl[7:] = self.ctrl.copy()
        # self.ctrl = np.array(msg.position)
    '''

    '''
    robot init 값: 
    [0.   0.   1.03 1.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
     0.   0.   0.   0.   0.   0.   0.  ]
    '''
    def timer_callback(self):
        self.joint_state_publish(self.data.qpos, self.data.qvel)
        if self.init:
            # self.data.qpos = np.zeros((21))
            # self.data.qpos[:7] = np.array([0,0,1.03, 0,0,0,1]) # first 7 : floating joint
            # self.data.qpos[7:14] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Left hand initial positions
            # self.data.qpos[14:21] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Right hand initial positions
            print(self.data.qpos)
            self.init = False
            # self.joint_state_callback_test()
        if self.ctrlFlag: # teleop_manager 를 통해서 joinstate 를 받을 예정
            print("Applying control:", self.ctrl)
            self.data.qpos[7:14] = self.ctrl[:7]
            self.data.qpos[14:21] = self.ctrl[7:]
            mujoco.mj_forward(self.model, self.data)

    '''
    def run(self):
        self.joint_state_publish(self.data.qpos, self.data.qvel)

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:   
            while rclpy.ok() and viewer.is_running():
                # Step the physics
                mujoco.mj_step(self.model, self.data)
                # viewer.render()

            # joint_msg = JointState()
            # joint_msg.header.stamp = self.get_clock().now().to_msg()
            # joint_msg.position = self.data.qpos[:self.model.njnt].tolist()
            # joint_msg.velocity = self.data.qvel[:self.model.njnt].tolist()
            # self.publisher.publish(joint_msg)
                # rclpy.spin_once(self, timeout_sec=0.0)
                # viewer.sync()
    '''

    def main(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while rclpy.ok() and viewer.is_running():
                rclpy.spin_once(self, timeout_sec=0.0)
                viewer.sync()

def main():
    rclpy.init()
    node = MujocoSimulationNode()
    # rclpy.spin(node)
    node.main()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
