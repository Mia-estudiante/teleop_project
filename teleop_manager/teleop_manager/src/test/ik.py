import numpy as np
import pinocchio as pin
from teleop_manager.src.robot.h1_2_wrapper import H12Wrapper

robot = H12Wrapper()
print(f"Current robot q: {robot.state.q}")
robot.computeAllTerms()
'''
state.l_oMi:   R =
1 0 0
0 1 0
0 0 1
  p =     0.232    0.2095 0.0949799
'''
print("state.l_oMi:", robot.state.l_oMi) 

'''
world 기준
'''
l_goal = pin.SE3(np.array([[1,0,0,0.432],
        [0,1,0,0.2095],
        [0,0,1,1.1549799],
        [0,0,0,1]]))
l_qdes = np.zeros(7)

# 1. 포즈 오차 계산 (목표와 현재의 차이)
l_dMi = robot.state.l_oMi.inverse() * l_goal # omi2m m2omi
# print("l_dMi:", l_dMi)
x_err_l = pin.log(l_dMi)
# print("x_err_l:", x_err_l)

# 2. 기본 공식 (Primary Task만 적용)
# qdot = J_inv * (Gain * error)
l_qdot = np.linalg.pinv(robot.state.l_J) @ (1 * x_err_l)

# 3. 적분하여 명령 생성
# print("l_qdes:", l_qdes)
# print("l_qdot:", l_qdot)
l_qdes += l_qdot * 0.01
r_qdes = np.zeros(7)
print("l_qdes:", l_qdes)
# print("l_qdes:", r_qdes)
# pin.forwardKinematics(robot.model, robot.data, np.concatenate((l_qdes, r_qdes)))
# pin.updateFramePlacements(robot.model, robot.data)

robot.computeAllTerms()

#2번째, 단 goal 동일
# 1. 포즈 오차 계산 (목표와 현재의 차이)
print("2번째 state.l_oMi:", robot.state.l_oMi) 
l_dMi = robot.state.l_oMi.inverse() * l_goal # omi2m m2omi
# print("l_dMi:", l_dMi)
x_err_l = pin.log(l_dMi)
# print("x_err_l:", x_err_l)

# 2. 기본 공식 (Primary Task만 적용)
# qdot = J_inv * (Gain * error)
l_qdot = np.linalg.pinv(robot.state.l_J) @ (1 * x_err_l)

# 3. 적분하여 명령 생성
l_qdes += l_qdot * 0.01
print("2번째 l_qdes:", l_qdes)
