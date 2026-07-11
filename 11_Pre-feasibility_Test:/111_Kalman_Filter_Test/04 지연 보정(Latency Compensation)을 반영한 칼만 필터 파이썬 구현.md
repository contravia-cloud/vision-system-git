# 지연 보정(Latency Compensation)을 반영한 칼만 필터 파이썬 구현

컨베이어 속도가 일정할 때 가장 효율적이고 실무에서 선호되는 **'방법 1: 지연 시간만큼 미래 예측(간이 보정법)'**을 반영한 파이썬 코드입니다. 

앞서 설계한 **12차원 상태 공간 모델**(로봇 위치/속도 6차원 + 타겟 위치/속도 6차원)을 기반으로 하며, 비전 데이터가 들어왔을 때 **연산 지연 시간(Latency)만큼 타겟의 위치를 앞으로 밀어서 업데이트**하는 로직을 포함하고 있습니다.

---

## 1. 파이썬 예제 코드

행렬 연산을 위해 `numpy` 라이브러리가 필요합니다. (`pip install numpy`)

```python
import time
import numpy as np


class LatencyCompensatedKF:

    def __init__(self, dt=0.01):
        """dt: 로봇의 제어 주기 (예: 100Hz = 0.01초)"""
        self.dt = dt

        # 1. 상태 벡터 (State Vector, X)
        # X = [x_r, y_r, z_r, vx_r, vy_r, vz_r, x_t, y_t, z_t, vx_t, vy_t, vz_t]^T (12x1)
        self.X = np.zeros((12, 1))

        # 2. 시스템 행렬 (State Transition Matrix, F)
        self.F = np.eye(12)
        for i in range(3):
            self.F[i, i + 3] = self.dt  # 로봇 위치 = 로봇 위치 + 로봇 속도 * dt
            self.F[i + 6, i + 9] = self.dt  # 타겟 위치 = 타겟 위치 + 타겟 속도 * dt

        # 3. 공분산 행렬 (P)
        self.P = np.eye(12) * 0.1

        # 4. 프로세스 노이즈 행렬 (Q)
        self.Q = np.eye(12)
        for i in range(6):
            self.Q[i, i] = 1e-6  # 로봇 모델은 매우 정확함
            self.Q[i + 6, i + 6] = 1e-4  # 타겟(컨베이어) 모델의 미세한 불확실성

        # 5. 측정 행렬 (Measurement Matrix, H)
        # 로봇 측정값(3차원), 카메라 상대 측정값(3차원) -> 총 6차원 측정
        self.H = np.zeros((6, 12))
        self.H[0:3, 0:3] = np.eye(3)  # 로봇 위치 계측
        self.H[3:6, 0:3] = -np.eye(3)  # 상대 위치 계측 (타겟 - 로봇)에서 -로봇 부분
        self.H[3:6, 6:9] = np.eye(3)  # 상대 위치 계측 (타겟 - 로봇)에서 +타겟 부분

        # 6. 측정 노이즈 행렬 (R)
        self.R = np.eye(6)
        for i in range(3):
            self.R[i, i] = 1e-6  # 로봇 인코더는 매우 정확함
            self.R[i + 3, i + 3] = 1e-3  # 비전 센서는 노이즈가 상대적으로 큼

    def init_state(self, robot_pos, target_pos, conveyor_vel):
        """초기 위치 및 컨베이어 속도 지정"""
        self.X[0:3] = np.array(robot_pos).reshape(3, 1)  # 로봇 초기 위치
        self.X[6:9] = np.array(target_pos).reshape(3, 1)  # 타겟 초기 위치
        self.X[9:12] = np.array(conveyor_vel).reshape(3, 1)  # 컨베이어 고정 속도

    def predict(self):
        """[Predict 단계] 매 제어 주기(예: 10ms)마다 로봇과 타겟의 다음 위치 예측"""
        self.X = np.dot(self.F, self.X)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.X

    def update_with_latency(self, measured_robot_pos, measured_relative_pos, latency):
        """[Update 단계] 비전 지연 시간(latency)을 고려하여 업데이트 수행

        measured_robot_pos: 비전 결과가 도착한 '현재'의 로봇 위치
        measured_relative_pos: 비전 알고리즘이 '과거' 셔터 오픈 시점에 계산한 상대 위치
        latency: 셔터 오픈 시점부터 현재까지 걸린 시간 (단위: 초)
        """
        # 현재 추정된 컨베이어(타겟)의 속도를 가져옴
        v_xt = self.X[9, 0]
        v_yt = self.X[10, 0]
        v_zt = self.X[11, 0]
        target_vel = np.array([v_xt, v_yt, v_zt])

        # [핵심] 과거에 측정된 상대 위치에 (컨베이어 속도 * 지연시간)을 더해 현재 시점의 값으로 보정
        compensated_relative