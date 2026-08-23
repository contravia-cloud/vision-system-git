# VLA 에이전트 개발 관련

<details>
<summary>AI 에이전트에서 지능(LLM)을 통해 추론을 한다고하고 자율성을 위해 React, CoT, ToT 등도 에이전트 구현시 넣으면 더 추론을 잘한다고 하던데 VLA 구성 시 라마는 React, CoT, ToT 이런거 없어도 될까?</summary>

<br>

> 아닙니다. VLA 구성을 하더라도 ReAct, CoT, ToT 같은 추론 프레임워크는 절대 빼놓을 수 없으며, 오히려 로봇 도메인에 맞게 변형된 형태로 반드시 포함해야 합니다.  
일반적인 텍스트 모델과 달리, VLA(Vision-Language-Action) 모델은 "단 한 번의 판단 착오가 로봇의 충돌이나 파손"이라는 치명적인 현실 물리 피해로 이어집니다. 라마(Llama) 백본이 아무리 지능이 높아도 시각 정보를 보고 곧바로 "모터 값(Action)"만 출력하는 엔드투엔드(End-to-End) 매핑 방식을 쓰면, 복잡하고 긴 단계를 거쳐야 하는 작업(Long-horizon Task)에서 100% 실패하게 됩니다. [1, 2] 
따라서 최근 Embodied AI(실체화된 AI) 학계에서는 텍스트 기반의 CoT나 ReAct를 물리 공간에 맞춘 '새로운 형태의 에이전트 추론 루프'로 진화시켜 적용하고 있습니다.
------------------------------
## 1. 텍스트 CoT의 한계와 VLA 특화 CoT의 등장
텍스트 환경에서 CoT(Chain of Thought)는 "1단계: 컵을 찾는다 -> 2단계: 손을 뻗는다"처럼 글로 생각을 정리하는 방식입니다. 하지만 로봇에게는 단어 '잡기'와 실제 '손가락 관절 모터의 좌표 제어' 사이에 엄청난 괴리(시맨틱-키네마틱 갭, Semantic-Kinematic Gap)가 존재합니다. [3, 4, 5, 6] 
이를 해결하기 위해 VLA 에이전트 설계 시 다음과 같은 변형된 CoT 기법들이 필수적으로 논문과 실제 서비스에 쓰이고 있습니다.

* Embodied CoT (ECoT): 행동을 출력하기 전에, 라마가 머릿속으로 [물체의 Bounding Box 좌표 생성] -> [서브 태스크 정의] -> [장애물 회피 경로 계산] 같은 시각적·물리적 단계를 텍스트 토큰과 좌표 토큰으로 먼저 추론한 뒤 최종 액션을 출력하게 만듭니다. 실제로 [OpenVLA 기반 연구(ECoT)](https://embodied-cot.github.io/)에 따르면, 이 추론 단계를 넣었을 때 로봇의 작업 성공률이 무려 28%나 상승했습니다. [7, 8] 
* Visual CoT: 최근 SOTA(최고 성능)를 기록한 [CoT-VLA](https://cot-vla.github.io/) 같은 모델은 행동하기 전에 "내가 이 행동을 하면 미래에 바뀔 방 안의 모습(Subgoal Image)"을 모델 내부에서 미리 상상(예측)하게 만듭니다. 머릿속으로 미래 결과 이미지를 먼저 그려본 뒤(CoT), 그 이미지에 도달하기 위한 모터 값을 출력하는 방식입니다. [2, 9, 10] 
* Action CoT (ACoT): 말로 생각하는 대신, 거칠고 대략적인 이동 궤적(Rough Trajectory Sketch)을 먼저 정방향으로 쭉 생성해 본 뒤, 이를 정밀한 모터 제어 값으로 디코딩하며 정제하는 행동 공간 내의 CoT 기법입니다. [5, 6] 

------------------------------
## 2. VLA 환경에서 ReAct와 ToT가 필요한 이유
VLA 구성 시 하드웨어 제어뿐만 아니라 '상황 인지 루프'가 결합되므로 ReAct와 ToT의 개념이 고스란히 녹아들어야 합니다.
## 💡 ReAct (Reason + Act) ──▶ "실시간 Closed-Loop 제어"
로봇이 물건을 집으려는데 미끄러져서 떨어뜨렸다고 가정해 보겠습니다.

* ReAct가 없다면: 모델은 이미 집었다고 착각하고 다음 단계(문 열기)로 넘어가 허공에 손짓을 합니다.
* ReAct가 있다면 (Thought-Action-Observation): [Thought: 컵을 집으려고 시도했다] -> [Action: 집기 명령 수행] -> [Observation: 카메라 시각 데이터 확인 결과 컵이 바닥에 그대로 있음] -> [Thought: 실패했으니 다시 시도해야 한다]라는 실시간 피드백 루프(Closed-Loop)를 돌릴 수 있게 됩니다. [11, 12] 

## 🌲 ToT (Tree of Thoughts) ──▶ "경로 탐색 및 시뮬레이션"
"싱크대에 있는 접시를 건조대로 옮기되, 가는 길에 있는 유리컵을 건드리지 마라"와 같은 복잡한 제약 조건이 걸린 작업의 경우입니다.

* 라마 백본이 여러 개의 가상 행동 경로(A안: 위로 돌아서 가기, B안: 오른쪽으로 피해 가기)를 트리 구조로 확장해가며 성공 확률과 위험도를 내부적으로 시뮬레이션(MCTS 등과 결합)하고, 가장 안전한 경로를 선택하는 ToT 방식의 아키텍처가 VLA의 안전성을 극대화합니다.

------------------------------
## 3. 개발 컨셉을 위한 최종 제언
질문하신 컨셉인 [Llama 백본 + VLA Action 에이전트]를 빌드하실 때, 구조는 다음과 같이 가져가시는 것이 정석입니다.

[입력: 이미지 + 명령텍스트] 
        │
        ▼
[1단계: 내부 추론 (CoT / ToT)] ──▶ "물체 위치 파악 및 이동 경로를 벡터/텍스트로 정리"
        │
        ▼
[2단계: 실행 및 출력 (Action)] ──▶ "최종 로봇 관절 모터 값(Action Token) 출력"
        │
        ▼
[3단계: 피드백 반영 (ReAct)]   ──▶ "카메라 화면으로 결과 보고 행동 수정" (반복)

따라서 라마의 기본 지능에만 의존해 바로 Action을 뽑지 마시고, 프롬프트나 파인튜닝 데이터셋 설계 단계에서 반드시 [생각 과정(CoT/ReAct 토큰) -> 행동(Action 토큰)] 순서로 모델이 출력하도록 유도하는 아키텍처를 설계하셔야 실패 없는 고성능 VLA 에이전트를 만드실 수 있습니다. [7, 8] 
이 설계와 관련해서 더 깊게 들어가 보고 싶은 부분이 있으신가요?

* 라마에게 행동 전 추론(CoT) 데이터셋을 학습시키는 구체적인 포맷이 궁금하신가요?
* 실시간 제어를 위해 추론 속도(Latency)를 줄여주는 표현 공간 추론(RDV-VLA 등) 방식에 대해 알고 싶으신가요? [13] 


[1] [https://arxiv.org](https://arxiv.org/abs/2503.22020)
[2] [https://arxiv.org](https://arxiv.org/html/2503.22020v1)
[3] [https://seohyun00.tistory.com](https://seohyun00.tistory.com/17)
[4] [https://arxiv.org](https://arxiv.org/html/2411.02093v1)
[5] [https://www.youtube.com](https://www.youtube.com/watch?v=43yUDoGKh9w)
[6] [https://www.youtube.com](https://www.youtube.com/watch?v=0dKcjbMRsLI&t=8)
[7] https://embodied-cot.github.io
[8] https://embodied-cot.github.io
[9] https://cot-vla.github.io
[10] [https://www.youtube.com](https://www.youtube.com/watch?v=jScOxG1_SE0&t=69)
[11] [https://www.youtube.com](https://www.youtube.com/watch?v=iNS7zaxWvJ8&vl=ko)
[12] [https://arxiv.org](https://arxiv.org/html/2508.17281v2)
[13] [https://www.youtube.com](https://www.youtube.com/watch?v=sx2ytqe3hcI)

</details>
<br>
<details>
<summary>
파이썬으로 AI Agent 만들기 (프레임워크 X)</summary>
<br>
https://youtu.be/3wk45Ow3m3M  

https://github.com/dabidstudio/python-ai-agents

1. 프롬프트 체이닝
2. 라우팅
3. 병렬처리
4. 오케스트레이터-워커
5. 평가-최적화

</details>