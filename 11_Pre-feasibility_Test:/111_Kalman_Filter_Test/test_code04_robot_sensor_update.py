# 가변 차원 멀티레이트 센서 업데이트 기반 비주얼 서보잉 시스템 설계

import sys
import time
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# [클래스] 가변 측정 모델을 지원하는 12차원 칼만 필터
# ==========================================
class VariableMeasurementKF:
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

        # 센서 기본 측정 노이즈 정의
        self.R_robot = np.eye(3) * 1e-6       # 로봇 엔코더 위치 노이즈
        self.R_vision = np.eye(3) * 1e-3      # 비전 센서 상대 위치 노이즈

    def init_state(self, robot_pos, target_pos, conveyor_vel):
        """초기 필터 상태 설정 (차원 붕괴 방지 보완)"""
        self.X[0:3] = np.array(robot_pos).reshape(3, 1)
        self.X[3:6] = np.zeros((3, 1))
        self.X[6:9] = np.array(target_pos).reshape(3, 1)
        self.X[9:12] = np.array(conveyor_vel).reshape(3, 1)

    # ------------------------------------------
    # [STEP 1] 예측 단계 (Predict)
    # ------------------------------------------
    def predict_step(self):
        """이전 상태 추정값(X)과 물리 법칙(F)만을 기반으로 독립 예측"""
        self.X = np.dot(self.F, self.X)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.X

    # ------------------------------------------
    # [STEP 2] 추정 / 보정 단계 (Update / Correct)
    # ------------------------------------------
    def update_step(self, measured_robot_pos, measured_relative_pos=None, latency=0.0):
        """센서 데이터의 유무에 따라 관측 모델(H, Z, R)의 차원을 동적으로 변경하여 추정"""
        
        # 1) 비전 데이터가 없을 때: 로봇 엔코더 데이터만으로 3차원 업데이트 진행
        if measured_relative_pos is None:
            Z = np.array(measured_robot_pos).reshape(3, 1)
            
            H = np.zeros((3, 12))
            H[0:3, 0:3] = np.eye(3)
            
            R = self.R_robot

        # 2) 비전 데이터가 있을 때: 로봇 위치 + 비전 상대 위치로 6차원 업데이트 진행
        else:
            v_xt, v_yt, v_zt = self.X[9, 0], self.X[10, 0], self.X[11, 0]
            target_vel = np.array([v_xt, v_yt, v_zt])

            # 지연 시간만큼 타겟이 더 움직였을 거리를 보정
            compensated_relative_pos = np.array(measured_relative_pos) + (target_vel * latency)

            Z = np.vstack((
                np.array(measured_robot_pos).reshape(3, 1),
                compensated_relative_pos.reshape(3, 1)
            ))
            
            H = np.zeros((6, 12))
            H[0:3, 0:3] = np.eye(3)
            H[3:6, 0:3] = -np.eye(3)
            H[3:6, 6:9] = np.eye(3)
            
            R = np.block([
                [self.R_robot, np.zeros((3, 3))],
                [np.zeros((3, 3)), self.R_vision]
            ])

        # 공통 칼만 필터 업데이트 공식
        S = np.dot(np.dot(H, self.P), H.T) + R
        K = np.dot(np.dot(self.P, H.T), np.linalg.inv(S))

        y = Z - np.dot(H, self.X)      
        self.X = self.X + np.dot(K, y)  

        I = np.eye(12)
        self.P = np.dot((I - np.dot(K, H)), self.P)
        return self.X


# ==========================================
# [클래스] 비주얼 서보잉 PI 제어기
# ==========================================
class VisualServoingController:
    def __init__(self, dt=0.01):
        self.dt = dt
        self.kp = np.array([3.0, 3.5, 3.0])
        self.ki_y = 4.5
        self.integral_error_y = 0.0
        self.max_integral_y = 0.1

    # ------------------------------------------
    # [STEP 3] 제어 단계 (Control)
    # ------------------------------------------
    def generate_control_command(self, estimated_target_pos, estimated_conveyor_vel, estimated_robot_pos):
        """정제된 '상태 추정값'들을 피드백 받아 제어 명령 생성"""
        pos_error = estimated_target_pos - estimated_robot_pos

        # X축 제어 (P + Feed-forward)
        cmd_vx = self.kp[0] * pos_error[0] + estimated_conveyor_vel[0]

        # Y축 제어 (PI)
        self.integral_error_y += pos_error[1] * self.dt
        self.integral_error_y = np.clip(
            self.integral_error_y, -self.max_integral_y / self.ki_y, self.max_integral_y / self.ki_y
        )
        cmd_vy = self.kp[1] * pos_error[1] + self.ki_y * self.integral_error_y

        # Z축 제어 (P)
        cmd_vz = self.kp[2] * pos_error[2]

        return np.array([cmd_vx, cmd_vy, cmd_vz])


# ==========================================
# 가상 실시간 환경 시뮬레이터 실행 함수
# ==========================================
def run_simulation():
    dt = 0.01
    total_steps = 300

    kf = VariableMeasurementKF(dt=dt)
    controller = VisualServoingController(dt=dt)

    # 실제 물리 상태 설정
    true_conveyor_vel = np.array([0.15, 0.0, 0.0])
    true_target_pos = np.array([0.1, 0.2, 0.0])
    true_robot_pos = np.array([0.0, 0.0, 0.4])

    # 초기 상태 주입 (np.array 형태로 안전하게 전달)
    kf.init_state(
        robot_pos=true_robot_pos,
        target_pos=true_target_pos + np.array([0.05, -0.08, 0.0]), # 의도적 초기 오차
        conveyor_vel=np.array([0.0, 0.0, 0.0])
    )

    vision_latency = 0.05
    vision_update_interval = 5

    history_true_target = []
    history_true_robot = []
    errors = []

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # --- 메인 루프 ---
    for step in range(total_steps):
        true_target_pos += true_conveyor_vel * dt

        # [STEP 1] 예측 단계
        kf.predict_step()

        # [STEP 2] 추정 / 보정 단계
        encoder_robot_pos = true_robot_pos + np.random.normal(0, 1e-4, 3) 

        if step >= vision_update_interval and step % vision_update_interval == 0:
            # 50ms 마다 비전 데이터가 들어오는 타이밍 (6차원 동시 퓨전 추정)
            past_idx = step - int(vision_latency / dt)
            measured_relative_past = (history_true_target[past_idx] - history_true_robot[past_idx]) + np.random.normal(0, 0.005, 3)
            
            kf.update_step(
                measured_robot_pos=encoder_robot_pos, 
                measured_relative_pos=measured_relative_past, 
                latency=vision_latency
            )
        else:
            # 평소 10ms 루프 타이밍 (로봇 데이터만으로 3차원 추정)
            kf.update_step(measured_robot_pos=encoder_robot_pos)

        # [STEP 3] 제어 단계
        estimated_robot_pos = kf.X[0:3].flatten()
        estimated_target_pos = kf.X[6:9].flatten()
        estimated_conveyor_vel = kf.X[9:12].flatten()

        robot_cmd_vel = controller.generate_control_command(
            estimated_target_pos, estimated_conveyor_vel, estimated_robot_pos
        )

        # 물리 하드웨어 한계 상황 시뮬레이션 (제어 효율 저하 및 미세 외란 주입)
        actual_robot_vel = robot_cmd_vel * 0.7 + np.array([0.0, -0.01, 0.0])
        true_robot_pos += actual_robot_vel * dt
        
        history_true_target.append(true_target_pos.copy())
        history_true_robot.append(true_robot_pos.copy())

        # 데이터 기록 및 시각화
        current_error = np.linalg.norm(true_target_pos - true_robot_pos)
        errors.append(current_error)

        if step % 2 == 0:
            ax1.cla()
            ax1.set_title("Real-time Variable Dimension Tracking (X-Y Plane)")
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
            ax2.set_title("Tracking Error (Every-step Encoder Update & Async Vision)")
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

