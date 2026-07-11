# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kalman Filter 기반 비주얼 서보잉 시뮬레이션 (지연 보정 + PI 제어기 + 실제 로봇 엔코더 속도 주입)
"""


import sys
import time
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# [클래스] 12차원 상태 공간 모델 기반 칼만 필터
# ==========================================
class LatencyCompensatedKF:
    def __init__(self, dt=0.01):
        self.dt = dt
        
        # 상태 벡터 X (12x1): [로봇 pos(3), 로봇 vel(3), 타겟 pos(3), 타겟 vel(3)]
        self.X = np.zeros((12, 1))

        # 시스템 전이 행렬 F (물리 법칙 행렬)
        self.F = np.eye(12)
        for i in range(3):
            self.F[i, i + 3] = self.dt       # 로봇 위치 = 로봇 위치 + 로봇 속도 * dt
            self.F[i + 6, i + 9] = self.dt   # 타겟 위치 = 타겟 위치 + 타겟 속도 * dt

        self.P = np.eye(12) * 0.1 # 추정 오차 공분산
        
        # 시스템 노이즈 Q
        self.Q = np.eye(12)
        for i in range(6):
            self.Q[i, i] = 1e-6
            self.Q[i + 6, i + 6] = 1e-4

        # 관측 행렬 H (측정 벡터 Z와 상태 벡터 X의 관계식)
        # Z = [로봇_실제_위치(3), 비전_상대_위치(3)]
        self.H = np.zeros((6, 12))
        self.H[0:3, 0:3] = np.eye(3)      # 로봇 위치 측정값 매칭
        self.H[3:6, 0:3] = -np.eye(3)     # 상대 위치 = 타겟 위치 - 로봇 위치
        self.H[3:6, 6:9] = np.eye(3)

        # 센서 측정 노이즈 R
        self.R = np.eye(6)
        for i in range(3):
            self.R[i, i] = 1e-6       # 로봇 엔코더 위치 노이즈 (매우 정밀함)
            self.R[i + 3, i + 3] = 1e-3  # 비전 센서 상대 위치 노이즈 (불확실성 높음)

    def init_state(self, robot_pos, target_pos, conveyor_vel):
        """초기 필터 상태 설정"""
        self.X[0:3] = np.array(robot_pos).reshape(3, 1)
        self.X[6:9] = np.array(target_pos).reshape(3, 1)
        self.X[9:12] = np.array(conveyor_vel).reshape(3, 1)

    # ------------------------------------------
    # [1] 예측 단계 (Predict)
    # ------------------------------------------
    def predict_step(self, actual_robot_vel):
        """
        [개선사항] 지령값(Command)이 아닌, 특이점 등이 반영된 '실제 로봇 속도(Encoder)'를 주입받아 예측합니다.
        """
        self.X[3:6] = np.array(actual_robot_vel).reshape(3, 1) # 실제 로봇 속도로 필터 동기화
        self.X = np.dot(self.F, self.X)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.X

    # ------------------------------------------
    # [2] 추정 / 보정 단계 (Update / Correct)
    # ------------------------------------------
    def update_step_with_vision(self, measured_robot_pos, measured_relative_pos, latency):
        """비전 센서 데이터가 들어왔을 때, 시간 지연을 보정하여 상태를 최적으로 추정(Update)합니다."""
        v_xt, v_yt, v_zt = self.X[9, 0], self.X[10, 0], self.X[11, 0]
        target_vel = np.array([v_xt, v_yt, v_zt])

        # 비전 지연 시간(latency)만큼 타겟이 더 움직였을 거리를 보정 (Time Latency Compensation)
        compensated_relative_pos = np.array(measured_relative_pos) + (target_vel * latency)

        # 측정 벡터 Z 구성 (로봇 엔코더 위치 피드백 + 지연 보정된 비전 데이터)
        Z = np.vstack((
            np.array(measured_robot_pos).reshape(3, 1),
            compensated_relative_pos.reshape(3, 1)
        ))

        # 칼만 이득(Kalman Gain) 및 상태 수정 연산
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        y = Z - np.dot(self.H, self.X) # 측정 잔차
        self.X = self.X + np.dot(K, y) # 최적 상태 추정 완료

        I = np.eye(12)
        self.P = np.dot((I - np.dot(K, self.H)), self.P)
        return self.X


# ==========================================
# [클래스] 비주얼 서보잉 PI 제어기
# ==========================================
class VisualServoingController:
    def __init__(self, dt=0.01):
        self.dt = dt
        # 제어 게인 설정
        self.kp = np.array([3.0, 3.5, 3.0])
        self.ki_y = 4.5
        self.integral_error_y = 0.0
        self.max_integral_y = 0.1 # 안티 윈드업 제한

    # ------------------------------------------
    # [3] 제어 단계 (Control)
    # ------------------------------------------
    def generate_control_command(self, estimated_target_pos, estimated_conveyor_vel, actual_robot_pos):
        """최신 추정 상태를 기반으로 로봇 TCP 속도 명령을 생성합니다."""
        # 현재 추정 상태 기준 로봇과의 오차 계산
        pos_error = estimated_target_pos - actual_robot_pos

        # X축 제어: P 제어 + 컨베이어 속도 피드포워드(Feed-forward)
        cmd_vx = self.kp[0] * pos_error[0] + estimated_conveyor_vel[0]

        # Y축 제어: PI 제어 (정지 상태축이므로 피드포워드 생략)
        self.integral_error_y += pos_error[1] * self.dt
        self.integral_error_y = np.clip(
            self.integral_error_y, -self.max_integral_y / self.ki_y, self.max_integral_y / self.ki_y
        )
        cmd_vy = self.kp[1] * pos_error[1] + self.ki_y * self.integral_error_y

        # Z축 제어: P 제어
        cmd_vz = self.kp[2] * pos_error[2]

        return np.array([cmd_vx, cmd_vy, cmd_vz])


# ==========================================
# 가상 실시간 환경 시뮬레이터 실행 함수
# ==========================================
def run_simulation():
    dt = 0.01  # 10ms 주기
    total_steps = 300

    # 객체 생성
    kf = LatencyCompensatedKF(dt=dt)
    controller = VisualServoingController(dt=dt)

    # 실제 물리 환경의 초기 상태 (우리가 모르는 절대 진실)
    true_conveyor_vel = np.array([0.15, 0.0, 0.0])
    true_target_pos = np.array([0.1, 0.2, 0.0])
    true_robot_pos = np.array([0.0, 0.0, 0.4])

    # 초기 필터 상태 유발 (일부러 오차를 주어 필터 성능 확인)
    kf.init_state(
        robot_pos=true_robot_pos,
        target_pos=true_target_pos + np.array([0.05, -0.08, 0.0]), # Y축 오차 -80mm 삽입
        conveyor_vel=[0.0, 0.0, 0.0]
    )

    # 비전 센서 하드웨어 제약 설정
    vision_latency = 0.05
    vision_update_interval = 5

    history_true_target = []
    history_true_robot = []
    errors = []

    # 그래프 셋업
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # --- 메인 실시간 제어 및 추정 루프 ---
    for step in range(total_steps):
        # 0. 실제 물리 타겟(엔진 볼트 구멍) 이동
        true_target_pos += true_conveyor_vel * dt

        # [필터에서 최신 데이터 추출]
        estimated_target_pos = kf.X[6:9].flatten()
        estimated_conveyor_vel = kf.X[9:12].flatten()

        # ------------------------------------------
        # [STEP 3] 제어 단계 (Control)
        # ------------------------------------------
        # 지령 속도(Command) 계산
        robot_cmd_vel = controller.generate_control_command(
            estimated_target_pos, estimated_conveyor_vel, true_robot_pos
        )

        # ------------------------------------------
        # [현실 하드웨어 제약 시뮬레이션] 
        # 질문자님이 지적하신 특이점, 조인트 한계 등으로 인한 출력 저하 구현
        # ------------------------------------------
        # 지령 속도의 70% 수준만 실제 출력되고, Y축으로 역기구학 오차가 미세하게 발생한다고 가정
        actual_robot_vel = robot_cmd_vel * 0.7 + np.array([0.0, -0.01, 0.0])
        true_robot_pos += actual_robot_vel * dt # 실제 로봇 엔코더 위치가 됨
        
        # 물리 히스토리 기록
        history_true_target.append(true_target_pos.copy())
        history_true_robot.append(true_robot_pos.copy())

        # ------------------------------------------
        # [STEP 1] 예측 단계 (Predict)
        # ------------------------------------------
        # 지령(robot_cmd_vel)이 아니라 로봇 엔코더에서 읽어온 '실제 속도(actual_robot_vel)'를 주입!
        kf.predict_step(actual_robot_vel=actual_robot_vel)

        # ------------------------------------------
        # [STEP 2] 추정 / 보정 단계 (Update / Correct)
        # ------------------------------------------
        if step >= vision_update_interval and step % vision_update_interval == 0:
            # 비전 카메라가 찍었던 지연 시점(50ms 전)의 인덱스 계산
            past_idx = step - int(vision_latency / dt)
            past_target = history_true_target[past_idx]
            past_robot = history_true_robot[past_idx]

            # 카메라 노이즈가 섞인 상대 위치 데이터 생성
            vision_noise = np.random.normal(0, 0.005, 3)
            measured_relative_past = (past_target - past_robot) + vision_noise

            # 비전이 들어오는 주기에만 타겟 상태를 업데이트(추정값 보정)
            # 이때도 '실제 로봇 TCP 엔코더 위치(true_robot_pos)'를 넘겨줍니다.
            kf.update_step_with_vision(
                measured_robot_pos=true_robot_pos,
                measured_relative_pos=measured_relative_past,
                latency=vision_latency
            )

        # 오차 모니터링 및 시각화 (동일)
        current_error = np.linalg.norm(true_target_pos - true_robot_pos)
        errors.append(current_error)

        if step % 2 == 0:
            ax1.cla()
            ax1.set_title("Real-time Visual Servoing Tracking (X-Y Plane)")
            ax1.set_xlabel("X Position (m)")
            ax1.set_ylabel("Y Position (m)")
            ax1.grid(True)
            ax1.set_xlim(-0.1, 1.0)
            ax1.set_ylim(-0.1, 0.5)

            ax1.scatter(true_target_pos[0], true_target_pos[1], color="red", s=100, label="Actual Target")
            ax1.scatter(true_robot_pos[0], true_robot_pos[1], color="blue", s=80, label="Robot TCP (Encoder)")
            ax1.scatter(estimated_target_pos[0], estimated_target_pos[1], color="green", marker="x", s=100, label="KF Estimated Target")

            target_history_np = np.array(history_true_target)
            robot_history_np = np.array(history_true_robot)
            ax1.plot(target_history_np[:, 0], target_history_np[:, 1], "r--", alpha=0.5)
            ax1.plot(robot_history_np[:, 0], robot_history_np[:, 1], "b-", alpha=0.5)
            ax1.legend(loc="upper left")

            ax2.cla()
            ax2.set_title("Tracking Error Over Time (Robust to Kinematic Loss)")
            ax2.set_xlabel("Time Step (10ms)")
            ax2.set_ylabel("Distance Error (m)")
            ax2.grid(True)
            ax2.plot(errors, color="purple", lw=2)
            ax2.set_ylim(0, 0.5)

            plt.draw()
            plt.pause(0.001)

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    run_simulation()