import os
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

class HandController(Node):
    def __init__(self):
        super().__init__('hand_controller_node')
        self.publisher = self.create_publisher(Float64MultiArray, '/unity/controller', 10)
        
        # 파일 경로 설정 (파일명이 동일한 디렉토리에 있다고 가정)
        lfingers_file_path = '/home/home/mujoco_ws/src/user_manager/user_manager/src/data/retargeted_joints_left_igris.txt'
        rfingers_file_path = '/home/home/mujoco_ws/src/user_manager/user_manager/src/data/retargeted_joints_right_igris.txt'

        if not os.path.exists(lfingers_file_path):
            self.get_logger().error(f"파일을 찾을 수 없습니다: {lfingers_file_path}")
            return
        if not os.path.exists(rfingers_file_path):
            self.get_logger().error(f"파일을 찾을 수 없습니다: {rfingers_file_path}")
            return

        with open(lfingers_file_path, 'r') as f:
            # 주석(#)이 있는 줄은 제외하고 데이터가 있는 줄만 리스트에 담기
            self.lfingers_lines = [line.strip() for line in f.readlines() 
                          if line.strip() and not line.startswith('#')]
        self.get_logger().info(f"총 {len(self.lfingers_lines)}개의 왼손 데이터 스텝을 로드했습니다.")
        self.lfingers_current_idx = 0

        with open(rfingers_file_path, 'r') as f:
            # 주석(#)이 있는 줄은 제외하고 데이터가 있는 줄만 리스트에 담기
            self.rfingers_lines = [line.strip() for line in f.readlines() 
                          if line.strip() and not line.startswith('#')]
        self.get_logger().info(f"총 {len(self.rfingers_lines)}개의 오른손 데이터 스텝을 로드했습니다.")
        self.rfingers_current_idx = 0

        # 20Hz (0.05초) 주기로 실행 (필요에 따라 조절)
        self.timer = self.create_timer(0.05, self.timer_callback)

    def timer_callback(self):
        if not self.lfingers_lines:
            return
        if not self.rfingers_lines:
            return

        if self.lfingers_current_idx >= len(self.lfingers_lines):
            self.lfingers_current_idx = 0  # 처음부터 다시 반복
        if self.rfingers_current_idx >= len(self.rfingers_lines):
            self.rfingers_current_idx = 0  # 처음부터 다시 반복
            
        try:
            # 1. 한 줄의 데이터를 가져와 실수형 리스트로 변환
            lfinger_joint_values = list(map(float, self.lfingers_lines[self.lfingers_current_idx].split()))
            rfinger_joint_values = list(map(float, self.rfingers_lines[self.rfingers_current_idx].split()))

            if len(lfinger_joint_values) >= 6 and len(rfinger_joint_values) >= 6:
                msg = Float64MultiArray()
                
                # 2. 26개 크기의 기본 리스트 생성 (모두 0.0으로 초기화)
                full_data = [0.0] * 12
                # full_data = [0.0] * 26
                
                # 3. 파이썬 리스트 상태에서 인덱스 7~12에 값 넣기
                # 순서: pinky -> ring -> middle -> index -> thumb_p -> thumb_y
                full_data[0:6] = lfinger_joint_values[:6]  # 왼손용
                full_data[6:] = rfinger_joint_values[:6]  # 오른손용
                # full_data[7:13] = lfinger_joint_values[:6]  # 왼손용
                # full_data[20:26] = rfinger_joint_values[:6]  # 오른손용

                # 4. 완성된 리스트를 통째로 msg.data에 할당
                msg.data = full_data
                
                # 5. 토픽 발행
                self.publisher.publish(msg)
                self.lfingers_current_idx += 1
                self.rfingers_current_idx += 1

        except Exception as e:
            self.get_logger().error(f"에러 발생: {e}")
            self.lfingers_current_idx += 1
            self.rfingers_current_idx += 1

def main(args=None):
    rclpy.init(args=args)
    node = HandController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()