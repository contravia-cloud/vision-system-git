import sys
import time
import matplotlib.pyplot as plt
import numpy as np

# [이전 단계에서 작성한 칼만 필터 클래스 포함]
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

        self.R = np.eye(6)
        for i in range(3):
            self.R[i, i] = 1e-6
            self.R[i + 3, i + 3] = 1e-2  # 노이즈 시각화를 위해 조금 키움

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


# --- 가상 환경 시뮬레이터 및 뷰어 ---
def run_simulation():
    dt = 0.01  # 로봇 제어 주기 (10ms)
    total_steps = 300  # 3초 동안 시뮬레이션 진행

    kf = LatencyCompensatedKF(dt=dt)

    # 1. 초기 물리 상태 정의 (가상 공간의 실제 참값들)
    true_conveyor_vel = np.array([0.15, 0.0, 0.0])  # 컨베이어가 X방향으로 0.15 m/s 기동
    true_target_pos = np.array(
        [0.1, 0.2, 0.0]
    )  # 엔진 볼트 구멍의 실제 초기 위치
    true_robot_pos = np.array([0.0, 0.0, 0.4])  # 로봇 TCP의 실제 초기 위치

    # 칼만 필터 초기화 (초기 타겟 위치는 약간의 오차가 있다고 가정)
    kf.init_state(
        robot_pos=true_robot_pos,
        target_pos=true_target_pos + np.array([0.05, -0.05, 0.0]),  # 오차 유발
        conveyor_vel=[0.0, 0.0, 0.0],  # 속도는 모르는 상태로 시작
    )

    # 2. 비전 카메라 지연 환경 설정
    vision_latency = 0.05  # 50ms (5스텝) 지연
    vision_update_interval = 5  # 50ms 마다 비전 결과 도달

    # 과거 상태를 임시 저장하기 위한 가상 큐 히스토리 (시뮬레이터용)
    history_true_target = []
    history_true_robot = []

    # 3. 실시간 뷰어 설정 (Matplotlib 상단/하단 2분할 뷰)
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # 상단: 2D 평면 위치 플롯
    ax1.set_title("Real-time Visual Servoing Tracking (X-Y Plane)")
    ax1.set_xlabel("X Position (m)")
    ax1.set_ylabel("Y Position (m)")
    ax1.grid(True)

    # 하단: 시간에 따른 오차(수렴 곡선) 플롯
    ax2.set_title("Tracking Error Converge")
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Error (m)")
    ax2.grid(True)

    errors = []

    # --- 메인 제어 루프 ---
    for step in range(total_steps):
        # [물리 시뮬레이션] 실제 엔진은 컨베이어를 타고 이동
        true_target_pos += true_conveyor_vel * dt

        # [비주얼 서보잉 로직] 로봇은 필터가 '추정'한 엔진 위치를 향해 단순 비례 제어로 쫓아감
        estimated_target_pos = kf.X[6:9].flatten()


        # 로봇 속도 제어 명령 (현재 로봇과 추정된 타겟 사이의 오차 기반)
        kp = 3.0  # 비례 이득
        # robot_cmd_vel = kp * (estimated_target_pos - true_robot_pos)

        # 1. 칼만 필터가 추정한 컨베이어의 속도를 가져옴
        estimated_conveyor_vel = kf.X[9:12].flatten()

        # 2. 오차 기반 제어에 '컨베이어 속도(피드포워드)'를 더해줌
        robot_cmd_vel = kp * (estimated_target_pos - true_robot_pos) + estimated_conveyor_vel




        # 로봇 실제 위치 업데이트
        true_robot_pos += robot_cmd_vel * dt

        # 가상 데이터 히스토리 저장 (비전 지연 재현용)
        history_true_target.append(true_target_pos.copy())
        history_true_robot.append(true_robot_pos.copy())

        # 칼만 필터 Predict 수행 (로봇 속도는 내부 측정값으로 전달되었다고 가정)
        kf.X[3:6] = robot_cmd_vel.reshape(3, 1)
        kf.predict()

        # [비전 업데이트 발생] 50ms 주기로 과거의 비전 결과 데이터 도달
        if step >= vision_update_interval and step % vision_update_interval == 0:
            # 50ms(5스텝) 전의 물리 상태 인덱스 계산
            past_idx = step - int(vision_latency / dt)

            past_target = history_true_target[past_idx]
            past_robot = history_true_robot[past_idx]

            # 50ms 전 과거 시점에 카메라가 촬영한 상대 위치 (노이즈 추가)
            noise = np.random.normal(0, 0.005, 3)  # 5mm 수준의 백색 잡음
            measured_relative_past = (past_target - past_robot) + noise

            # 지연 보정 기능이 탑재된 Update 수행
            kf.update_with_latency(
                measured_robot_pos=true_robot_pos,
                measured_relative_pos=measured_relative_past,
                latency=vision_latency,
            )

        # 실제 타겟과 실제 로봇 사이의 현재 거리 오차 계산
        current_error = np.linalg.norm(true_target_pos - true_robot_pos)
        errors.append(current_error)

        # --- 실시간 화면 갱신 (애니메이션) ---
        if step % 2 == 0:  # 20ms 마다 화면을 다시 그림 (성능 최적화)
            ax1.cla()
            ax1.set_title("Real-time Visual Servoing Tracking (X-Y Plane)")
            ax1.set_xlabel("X Position (m)")
            ax1.set_ylabel("Y Position (m)")
            ax1.grid(True)
            ax1.set_xlim(-0.1, 1.0)
            ax1.set_ylim(-0.1, 0.5)

            # 점으로 현재 상태 표현
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

            # 추적 궤적선 그리기
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

            # 하단 에러 그래프 갱신
            ax2.cla()
            ax2.set_title("Tracking Error Over Time")
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