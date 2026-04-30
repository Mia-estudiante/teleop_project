import os
import sys, argparse

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState, Image

import mujoco
import mujoco.viewer

import numpy as np
from cv_bridge import CvBridge

from ament_index_python.packages import get_package_share_directory

bridge = CvBridge()


class MujocoSimulationIgrisCTestNode(Node):
    def __init__(self, xml_file_name='igris_c_v2.xml'):
        super().__init__('mujoco_simulation_igris_c_test_node')
        self.get_logger().info(f'XML file: {xml_file_name}')

        descriptions_path = get_package_share_directory('igris_c_description')
        igrisc_xml_path = os.path.join(descriptions_path, xml_file_name)

        self.model = mujoco.MjModel.from_xml_path(igrisc_xml_path)
        if self.model is None:
            self.get_logger().error(f'Failed to load model from {igrisc_xml_path}')
            return
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=480, width=640)

        # === Reset to keyframe (handle name differences) ===
        # cube xml uses "home"/"pregrasp", non-cube uses "initial"
        keyframe_name = None
        for candidate in ["home", "initial", "pregrasp"]:
            kid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, candidate)
            if kid >= 0:
                keyframe_name = candidate
                mujoco.mj_resetDataKeyframe(self.model, self.data, kid)
                self.get_logger().info(f"Reset to keyframe '{candidate}' (id={kid})")
                break
        if keyframe_name is None:
            self.get_logger().warn("No keyframe found, using zero pose")

        # Forward to compute kinematics so qpos is consistent
        mujoco.mj_forward(self.model, self.data)

        # === QoS ===
        ctrl_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.publisher = self.create_publisher(JointState, '/mujoco/joint_states', 10)
        self.camera_publisher = self.create_publisher(Image, '/mujoco/camera', 10)
        self.subscriber = self.create_subscription(
            Float64MultiArray, '/mujoco/controller', self.joint_state_callback, ctrl_qos
        )
        self.timer = self.create_timer(0.01, self.timer_callback)

        # === Use model.nu (actual actuator count), not derived from njnt ===
        nu = self.model.nu
        self.get_logger().info(f"Model has {nu} actuators")
        self.ctrl        = np.zeros(nu)
        self.target_ctrl = np.zeros(nu)

        # CRITICAL: initialize smooth_ctrl to current arm pose, not zeros.
        # qpos layout: [floating_base 7 | left_arm 7 | right_arm 7 | (cube 7?)]
        # The 14 arm angles start at qpos index 7.
        self.smooth_ctrl = self.data.qpos[7:7 + nu].copy()

        self.get_logger().info(
            f"Initial smooth_ctrl from keyframe: {self.smooth_ctrl}"
        )

        self.init = True
        self.ctrlFlag = False
        self.alpha = 0.25  # was 0.05 — too slow. 0.25 → time constant ~36ms

    def joint_state_publish(self, qpos, qvel):
        joint_msg = JointState()
        # qpos[7:14]=left arm, qpos[14:21]=right arm
        # qvel[6:13]=left arm, qvel[13:20]=right arm  (freejoint vel takes 6, not 7)
        left_qpos  = qpos[7:14].tolist()
        left_qvel  = qvel[6:13].tolist()
        right_qpos = qpos[14:21].tolist()
        right_qvel = qvel[13:20].tolist()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.position = left_qpos + right_qpos
        joint_msg.velocity = left_qvel + right_qvel
        self.publisher.publish(joint_msg)

    def camera_publish(self, camera_name='eef_camera'):
        self.renderer.update_scene(self.data, camera=camera_name)
        pixels = self.renderer.render()
        self.camera_publisher.publish(bridge.cv2_to_imgmsg(pixels))

    def joint_state_callback(self, msg: Float64MultiArray):
        n = min(len(msg.data), len(self.target_ctrl))
        for i in range(n):
            self.target_ctrl[i] = msg.data[i]

        # On the very first command, snap smooth_ctrl to it to avoid PD jump
        if not self.ctrlFlag:
            self.smooth_ctrl[:] = self.target_ctrl[:]
            self.ctrl[:] = self.target_ctrl[:]
            self.get_logger().info(
                f"First command received, smooth_ctrl snapped: {self.smooth_ctrl}"
            )

        self.ctrlFlag = True

    def timer_callback(self):
        self.joint_state_publish(self.data.qpos, self.data.qvel)

        if self.init:
            self.get_logger().info(f"Initial qpos: {self.data.qpos[:21]}")
            self.init = False

        # 0.01초 wall clock = 시뮬 0.01초가 되도록 substep
        n_sub = max(1, int(round(0.01 / self.model.opt.timestep)))  # = 20

        if self.ctrlFlag:
            self.smooth_ctrl = (1 - self.alpha) * self.smooth_ctrl + self.alpha * self.target_ctrl
            self.data.ctrl[:] = self.smooth_ctrl
            for _ in range(n_sub):
                mujoco.mj_step(self.model, self.data)

    def main(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while rclpy.ok() and viewer.is_running():
                rclpy.spin_once(self, timeout_sec=0.0)
                viewer.sync()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--xml', type=str, default='igris_c_v2.xml',
                        help='MuJoCo XML file name')
    args, unknown = parser.parse_known_args()
    rclpy.init(args=unknown)

    node = MujocoSimulationIgrisCTestNode(xml_file_name=args.xml)
    node.main()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()