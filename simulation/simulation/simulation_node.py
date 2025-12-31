import os

import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64MultiArray
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

        '''
        # 1. 모델에 정의된 '기본(Reference)' 포즈 확인
        print("Default qpos:", self.model.qpos0)
        # 2. 현재 데이터의 관절 위치 (초기화 직후)
        print("Initial qpos:", self.data.qpos)
        '''

        self.publisher = self.create_publisher(JointState, '/mujoco/joint_states', 10)
        self.subscriber = self.create_subscription(Float64MultiArray, '/mujoco/controller', self.joint_state_callback, 10)
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

    def joint_state_callback(self, msg: Float64MultiArray):
        for i in range(len(msg.data)):
            self.ctrl[i] = msg.data[i]
        self.ctrlFlag = True

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
            self.init = False
        if self.ctrlFlag: # teleop_manager 를 통해서 joinstate 를 받을 예정
            self.data.qpos[7:14] = self.ctrl[:7]
            self.data.qpos[14:21] = self.ctrl[7:]
            mujoco.mj_forward(self.model, self.data)

    def main(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while rclpy.ok() and viewer.is_running():
                rclpy.spin_once(self, timeout_sec=0.0)
                viewer.sync()

def main():
    rclpy.init()
    node = MujocoSimulationNode()
    node.main()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
