import numpy as np
import pinocchio as pin
from cyipopt import minimize_ipopt

class IPOPT_Solver:
    def __init__(self, model, data, frame_id):
        self.model = model
        self.data = data
        self.frame_id = frame_id
        
        # 관절 한계 설정 (Pinocchio 모델에서 가져옴)
        self.q_min = self.model.lowerPositionLimit
        print(f"self.q_min, {self.q_min}")
        self.q_max = self.model.upperPositionLimit

    def objective(self, q, target_M):
        """최소화할 목표: 현재 포즈와 목표 포즈 사이의 거리(에러)"""
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        
        # 현재 프레임의 위치/자세
        current_M = self.data.oMf[self.frame_id]
        
        # 포즈 에러 계산 (SE3 space)
        error_dist = pin.log(current_M.inverse() * target_M).vector
        
        # 에러의 제곱합 (L2 Norm squared)
        return 0.5 * np.sum(error_dist**2)

    def solve(self, q_guess, target_M):
        # 관절 한계 제약 조건 설정
        bounds = list(zip(self.q_min, self.q_max))
        
        # IPOPT 옵션 설정 (실시간성을 위해 반복 횟수 제한 가능)
        options = {
            'max_iter': 50,
            'tol': 1e-4,
            'print_level': 0,  # 로그 출력 안 함
        }
        
        # 최적화 실행
        res = minimize_ipopt(
            self.objective, 
            x0=q_guess, 
            args=(target_M,), 
            bounds=bounds,
            options=options
        )
        
        return res.x

# --- 실제 적용 예시 (avp_node 내부) ---

# 1. 초기화 (한 번만 수행)
# L_FRAME_ID = robot.model.getFrameId("left_wrist_frame")
# solver_l = IKSolverIPOPT(robot.model, robot.data, L_FRAME_ID)

# 2. 루프 내부에서 사용
# headMl_target = self.head_goal.inverse() * self.l_goal
# target_M_world = self.robot.state.head_oMi * headMl_target # 월드 좌표계 기준 목표

# 최적의 q를 직접 계산
# self.l_qdes = solver_l.solve(self.l_qdes, target_M_world)