import rclpy
from rclpy.node import Node
import numpy as np
import json
import pinocchio as pin
from geometry_msgs.msg import Pose, PoseArray

class AVPHandLoopPlaybackNode(Node):
    def __init__(self):
        super().__init__('avp_hand_loop_playback_node')
        
        # 1. 파일 경로 설정 (반복용으로 생성한 파일 경로로 수정하세요)
        # 만약 filtered_lfingers_avp.json을 그대로 쓴다면 코드 내에서 루프를 만듭니다.
        self.lfingers_json_path = '/home/home/mujoco_ws/src/user_manager/user_manager/src/data/v_sign_lfingers_avp.json'
        self.lfingers_publisher = self.create_publisher(PoseArray, '/lfingers', 10)
        
        # 2. 데이터 로드 및 루프 구성
        self.playback_data = self.load_and_create_loop()
        self.current_frame = 0
        self.total_frames = len(self.playback_data)
        
        if self.total_frames == 0:
            self.get_logger().error("재생할 데이터가 없습니다!")
        else:
            self.get_logger().info(f"{self.total_frames} 프레임 루프 데이터를 로드했습니다. 무한 재생을 시작합니다.")

        # 3. 타이머 설정 (100Hz)
        self.timer = self.create_timer(0.01, self.timer_callback)

    def load_and_create_loop(self):
        try:
            with open(self.lfingers_json_path, 'r') as f:
                raw_json = json.load(f)
            
            # 리스트를 numpy array로 변환
            data = [np.array(frame["data"]) for frame in raw_json]
            
            # 만약 로드한 파일이 이미 반복용이 아니라면 여기서 역재생을 붙여 루프를 만듭니다.
            # (이미 repeat_lfingers_avp.json을 만들었다면 아래 과정은 생략 가능합니다)
            # if "repeat" not in self.lfingers_json_path:
            #     backward = data[::-1][1:-1] # 첫/끝 중복 방지
            #     data = data + backward
                
            return data
        except Exception as e:
            self.get_logger().error(f"파일 로드 실패: {e}")
            return []

    def timer_callback(self):
        if self.total_frames == 0:
            return
            
        # 나머지 연산자(%)를 사용하여 인덱스가 0 -> 1 -> ... -> max -> 0 으로 무한 순환하게 함
        idx = self.current_frame % self.total_frames
        
        lfingers_data = self.playback_data[idx]
        self.publish_lfingers(lfingers_data)
        
        self.current_frame += 1
        
        # 로그가 너무 많이 찍히는 것을 방지하기 위해 한 바퀴 돌 때마다 출력
        if idx == 0 and self.current_frame > 0:
            self.get_logger().info("루프 재생 중 (새 사이클 시작)")

    def publish_lfingers(self, lfingers_data):
        fingers_msg = PoseArray()
        fingers_msg.header.stamp = self.get_clock().now().to_msg()
        fingers_msg.header.frame_id = "world"

        for i in range(len(lfingers_data)):
            se3 = pin.SE3(lfingers_data[i])
            xyzquat = pin.SE3ToXYZQUAT(se3)
            
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = xyzquat[:3]
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = xyzquat[3:]
            fingers_msg.poses.append(pose)
            
        self.lfingers_publisher.publish(fingers_msg)

def main():
    rclpy.init()
    node = AVPHandLoopPlaybackNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('재생 중단')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()