import sys
import time
import matplotlib.pyplot as plt
import numpy as np


# [12차원 상태 공간 모델 기반 칼만 필터 클래스]
class LatencyCompensatedKF:

    def __init__(self, dt=0.01):
        self.dt = dt
        self.X = np.zeros((12, 1))

        self.F = np.eye(12)
        for i in range(3):
            self.F[i, i + 3] = self.dt
            self.F[i + 6, i + 9] = self.dt

        self.P = np.eye(12) * 0.1

        self.Q = np.eye(12)
        for i in range(6):
            self.Q[i, i] = 1e-6
            self.Q[i + 6, i + 6] = 1e-4

        self.H = np.zeros((6, 12))
        self.H[0:3, 0:3] = np.eye(3)
        self.H[3:6, 0:3] = -np.eye(3)
        self.H[3:6, 6:9] = np.eye(3)

        # 튜닝 포인트 1: Y축 수렴 성능을 올리기 위해 비전 R 노이즈 중 Y축 영역(인덱스 4)을 미세하게 조절 가능
        self.R = np.eye(6)
        for i in range(3):
            self.R[i, i] = 1e-6  # 로봇 인코더 노이즈
            self.R[i + 3, i + 3] = 1e-3  # 비전 센서 노이즈 기본값


    def init_state(self, robot_pos, target_pos, conveyor_vel):
        self.X[0:3] = np.array(robot_pos).reshape(3, 1)
        self.X[6:9] = np.array(target_pos).reshape(3, 1)
        self.X[9:12] = np.array(conveyor_vel).reshape(3, 1)

    def predict(self):
        self.X = np.dot(self.F, self.X)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.X

    def update_with_latency(self, measured_robot_pos, measured_relative_pos, latency):
        v_xt = self.X[9, 0]
        v_yt = self.X[10, 0]
        v_zt = self.X[11, 0]
        target_vel = np.array([v_xt, v_yt, v_zt])

        compensated_relative_pos = np.array(measured_relative_pos) + (
            target_vel * latency
        )

        Z = np.vstack(
            (
                np.array(measured_robot_pos).reshape(3, 1),
                compensated_relative_pos.reshape(3, 1),
            )
        )

        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        y = Z - np.dot(self.H, self.X)
        self.X = self.X + np.dot(K, y)

        I = np.eye(12)
        self.P = np.dot((I - np.dot(K, self.H)), self.P)
        return self.X


# --- 가상 환경 시뮬레이터 및 뷰어 실행 함수 ---
def run_simulation():
    dt = 0.01  # 로봇 제어 주기 (10ms)
    total_steps = 300  # 3초 동안 시뮬레이션 구동

    kf = LatencyCompensatedKF(dt=dt)

    # 1. 초기 물리 상태 정의
    true_conveyor_vel = np.array([0.15, 0.0, 0.0])  # 컨베이어 속도 (X방향)
    true_target_pos = np.array([0.1, 0.2, 0.0])  # 엔진 볼트 구멍 실제 초기 위치
    true_robot_pos = np.array([0.0, 0.0, 0.4])  # 로봇 TCP 실제 초기 위치

    # 초기 오차 유발 (Y방향 오차 유발 강조)
    kf.init_state(
        robot_pos=true_robot_pos,
        target_pos=true_target_pos + np.array([0.05, -0.08, 0.0]),  # Y축 오차 -80mm 삽입
        conveyor_vel=[0.0, 0.0, 0.0],
    )

    # 2. 비전 카메라 연산 지연 설정
    vision_latency = 0.05
    vision_update_interval = 5

    history_true_target = []
    history_true_robot = []

    # ----------------------------------------------------
    # 튜닝 포인트 2: PI 제어기 게인 설정 및 적분 누적기 선언
    # ----------------------------------------------------
    kp_x = 3.0
    kp_y = 3.5  # Y축 수렴 속도를 위해 비례 게인을 X보다 크게 설정 #4
    kp_z = 3.0

    ki_y = 4.5  # Y축 잔류 오차 제거를 위한 적분 게인 추가 #1.5
    integral_error_y = 0.0  # Y축 적분 오차 누적 버퍼
    max_integral_y = 0.1  # 안티 윈드업 제한 (최대 0.1 m/s 만큼만 누적 제어 보정하도록 제한)
    # ----------------------------------------------------

    # 3. 실시간 뷰어 그래픽 설정
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.set_title("Real-time Visual Servoing Tracking (X-Y Plane)")
    ax1.set_xlabel("X Position (m)")
    ax1.set_ylabel("Y Position (m)")
    ax1.grid(True)

    ax2.set_title("Tracking Error Converge")
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Error (m)")
    ax2.grid(True)

    errors = []

    # --- 실시간 메인 제어 루프 가동 ---
    for step in range(total_steps):
        true_target_pos += true_conveyor_vel * dt

        # 칼만 필터 추정 위치 및 컨베이어 속도 추출
        estimated_target_pos = kf.X[6:9].flatten()
        estimated_conveyor_vel = kf.X[9:12].flatten()

        # 현재 추정 상태 기준 로봇과의 오차 계산
        pos_error = estimated_target_pos - true_robot_pos

        # [X축 제어] 기존과 동일: P 제어 + 피드포워드
        robot_cmd_vx = kp_x * pos_error[0] + estimated_conveyor_vel[0]

        # [Y축 제어] 수정 가해짐: PI 제어 도입 (정지 상태축이므로 피드포워드는 생략 또는 0)
        # 1) 오차 적분 누적 (시간 간격 dt를 곱함)
        integral_error_y += pos_error[1] * dt
        # 2) 안티 윈드업 (Anti-windup): 적분항 포화 제한
        integral_error_y = np.clip(
            integral_error_y, -max_integral_y / ki_y, max_integral_y / ki_y
        )
        # 3) PI 제어 속도 명령 생성
        robot_cmd_vy = kp_y * pos_error[1] + ki_y * integral_error_y

        # [Z축 제어] 기존과 동일: P 제어
        robot_cmd_vz = kp_z * pos_error[2]

        # 제어 명령 결합 및 로봇 기구학 업데이트
        robot_cmd_vel = np.array([robot_cmd_vx, robot_cmd_vy, robot_cmd_vz])
        true_robot_pos += robot_cmd_vel * dt

        # 버퍼 저장
        history_true_target.append(true_target_pos.copy())
        history_true_robot.append(true_robot_pos.copy())

        # 칼만 필터 예측 단계
        kf.X[3:6] = robot_cmd_vel.reshape(3, 1)
        kf.predict()

        # 비전 업데이트 발생
        if step >= vision_update_interval and step % vision_update_interval == 0:
            past_idx = step - int(vision_latency / dt)
            past_target = history_true_target[past_idx]
            past_robot = history_true_robot[past_idx]

            noise = np.random.normal(0, 0.005, 3)
            measured_relative_past = (past_target - past_robot) + noise

            kf.update_with_latency(
                measured_robot_pos=true_robot_pos,
                measured_relative_pos=measured_relative_past,
                latency=vision_latency,
            )

        current_error = np.linalg.norm(true_target_pos - true_robot_pos)
        errors.append(current_error)

        # 화면 갱신 (20ms)
        if step % 2 == 0:
            ax1.cla()
            ax1.set_title("Real-time Visual Servoing Tracking (X-Y Plane)")
            ax1.set_xlabel("X Position (m)")
            ax1.set_ylabel("Y Position (m)")
            ax1.grid(True)
            ax1.set_xlim(-0.1, 1.0)
            ax1.set_ylim(-0.1, 0.5)

            ax1.scatter(
                true_target_pos[0],
                true_target_pos[1],
                color="red",
                s=100,
                label="Actual Target (Engine)",
            )
            ax1.scatter(
                true_robot_pos[0],
                true_robot_pos[1],
                color="blue",
                s=80,
                label="Robot TCP",
            )
            ax1.scatter(
                estimated_target_pos[0],
                estimated_target_pos[1],
                color="green",
                marker="x",
                s=100,
                label="KF Estimated Target",
            )

            target_history_np = np.array(history_true_target)
            robot_history_np = np.array(history_true_robot)
            ax1.plot(
                target_history_np[:, 0],
                target_history_np[:, 1],
                "r--",
                alpha=0.5,
            )
            ax1.plot(
                robot_history_np[:, 0],
                robot_history_np[:, 1],
                "b-",
                alpha=0.5,
            )
            ax1.legend(loc="upper left")

            ax2.cla()
            ax2.set_title("Tracking Error Over Time (with Y-axis PI Control)")
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