servoJ 명령을 사용해 LLM(Model) ➡️ MCP Server ➡️ MCP Client ➡️ 로봇 제어 시스템을 구축하려면, 실시간성을 보장하는 통신 구조를 짜는 것이 핵심입니다.
servoJ는 유니버설 로봇(UR) 등에서 약 2ms~8ms 주기로 연속적인 관절 각도 타겟을 주입해야 하는 실시간 모션 명령입니다. LLM의 텍스트 생성 속도(초 단위)로는 이 주기를 맞출 수 없으므로, LLM이 목표 포즈(혹은 궤적 방정식)를 생성하면, MCP 서버가 이를 받아 실시간 스트리밍(Trajectory Generation)하는 구조로 설계해야 합니다.
아래는 Anthropic의 Model Context Protocol (MCP) 아키텍처를 기반으로 구현하는 구체적인 방법입니다.
------------------------------
## 전체 시스템 아키텍처 구조

[ LLM (클라우드/로컬) ] 
      │ (텍스트 명령 / 자연어)
[ MCP Client (예: Claude Desktop / 커스텀 파이썬 앱) ]
      │ (JSON-RPC 기반 Tools 호출: "move_robot_servo")
[ MCP Server (파이썬 기반 비동기 서버) ]
      │ (URScript 소켓 통신: 125Hz / 500Hz 주기 스트리밍)
[ 실제 로봇 (UR RTDE 통신 환경) ]

------------------------------
## Step 1. MCP Server 구현 (Python)
MCP 서버는 LLM이 이해하고 호출할 수 있는 Tool을 정의하고, 클라이언트로부터 받은 최종 목적지 관절 각도(q)를 로봇에 servoJ 주기에 맞춰 쪼개서 보내는 역할을 합니다. [1] 
파이썬의 mcp 라이브러리와 로봇 RTDE 통신을 위한 ur_rtde 라이브러리를 사용해 작성합니다.

import asynciofrom mcp.server import Server, NotificationOptionsfrom mcp.server.models import InitializationOptionsimport mcp.types as typesimport rtde_control
# 1. MCP 서버 및 로봇 컨트롤러 초기화 (로봇 IP 입력)server = Server("robot-servo-controller")# 실제 환경에 맞춰 로봇 IP 주소를 입력하세요.ROBOT_IP = "192.168.1.100" rtde_c = None
try:
    rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
    print("로봇 RTDE 컨트롤러 연결 성공")except Exception as e:
    print(f"로봇 연결 실패(시뮬레이션 모드 또는 IP 확인 필요): {e}")
# 2. LLM에게 노출할 Tool 정의
@server.list_tools()async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="move_robot_servo",
            description="6자유도 로봇의 6개 관절 각도를 servoJ를 이용해 실시간 궤적으로 안전하게 이동시킵니다.",
            inputSchema={
                "type": "object",
                "properties": {
                    "target_joints": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "이동할 목표 관절 각도 리스트 [q0, q1, q2, q3, q4, q5] (라디안 단위)",
                        "minItems": 6,
                        "maxItems": 6
                    },
                    "duration": {
                        "type": "number",
                        "description": "목표 위치까지 도달할 총 시간 (초 단위, 기본값 3.0)",
                        "default": 3.0
                    }
                },
                "required": ["target_joints"]
            }
        )
    ]
# 3. Tool 실행 로직 (servoJ 루프 구현)
@server.call_tool()async def handle_call_tool(name: str, arguments: dict | None) -> list[types.TextContent]:
    if name != "move_robot_servo":
        raise ValueError(f"Unknown tool: {name}")
        
    if not arguments or "target_joints" not in arguments:
        return [types.TextContent(type="text", text="오류: target_joints 인자가 필요합니다.")]

    target_q = arguments["target_joints"]
    duration = arguments.get("duration", 3.0)

    if rtde_c is None:
        return [types.TextContent(type="text", text=f"로봇이 연결되지 않았습니다. 요청된 목표 각도: {target_q}")]

    # servoJ 실시간 스트리밍 제어 루프
    try:
        # 현재 관절 위치 가져오기
        start_q = rtde_c.getActualQ()
        
        # UR 로봇의 제어 주기 설정 (e.g., UR3/5/10은 125Hz = 0.008초, e-Series는 500Hz = 0.002초)
        dt = 0.008  
        steps = int(duration / dt)
        
        # 비동기 루프 내에서 실시간성 유지를 위해 스레드 풀 또는 태스크 분리 고려 가능
        for i in range(steps):
            # 선형 보간(Linear Interpolation)으로 매 주기마다 갈 중간 목표 각도 계산
            alpha = i / steps
            current_target_q = [start_q[j] + alpha * (target_q[j] - start_q[j]) for j in range(6)]
            
            # servoJ(q, 속도, 가속도, 시간t, 룩어헤드시간, 게인)
            # 여기서는 제어 주기 dt 동안만 움직이도록 t=dt 설정
            rtde_c.servoJ(current_target_q, 0.0, 0.0, dt, 0.1, 300)
            
            # 8ms 주기를 맞추기 위한 정밀 슬립
            await asyncio.sleep(dt)
            
        # 모션 종료 후 서보 모드 안전하게 해제
        rtde_c.servoStop()
        return [types.TextContent(type="text", text=f"성공: 로봇이 목표 각도 {target_q}로 servoJ 이동을 완료했습니다.")]

    except Exception as e:
        if rtde_c:
            rtde_c.servoStop()
        return [types.TextContent(type="text", text=f"로봇 제어 중 오류 발생: {str(e)}")]
# 4. 서버 메인 실행부async def main():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="robot-servo-controller",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )
if __name__ == "__main__":
    asyncio.run(main())

------------------------------
## Step 2. MCP Client 및 LLM 연결 설정
작성한 MCP 서버를 LLM 시스템(예: Claude Desktop 또는 파이썬 커스텀 클라이언트)에 등록해야 합니다. [2] 
## 방법 A: Claude Desktop 앱을 클라이언트로 쓰는 경우
claude_desktop_config.json 설정 파일에 방금 만든 파이썬 서버 스크립트를 등록합니다.

* 
* Windows 경로: %APPDATA%\Claude\claude_desktop_config.json
* Mac 경로: ~/Library/Application Support/Claude/claude_desktop_config.json [3, 4] 
* 

{
  "mcpServers": {
    "robot_servo_control": {
      "command": "python",
      "args": [
        "absolute/path/to/your/mcp_server.py"
      ]
    }
  }
}

*설정을 저장하고 Claude Desktop을 재시작하면, 채팅창 오른쪽 아래에 **망치 아이콘(Tools)*이 활성화되며 로봇 제어 기능이 주입됩니다.
## 방법 B: 직접 Python 코드로 MCP Client + LLM(API) 파이프라인 구축할 경우
Anthropic API를 호출할 때 MCP 서버의 스키마를 tools 인자에 주입하고, LLM이 tool_use를 출력하면 클라이언트가 MCP 서버의 handle_call_tool을 실행한 뒤 결과를 다시 LLM에 전달하는 루프 코드를 작성하시면 됩니다.
------------------------------
## Step 3. LLM에게 명령 내리기 (프롬프트 구성)
시스템이 연결되면 LLM에게 페르소나와 제어 규칙(안전 제약 조건)을 담은 시스템 프롬프트를 제공하는 것이 안전상 매우 중요합니다.
LLM 주입용 시스템 프롬프트 예시:

"너는 6자유도 협동로봇을 제어하는 인공지능 엔지니어다. 유저가 자연어로 명령을 내리면 상황에 맞는 각도(라디안 단위)를 계산해 move_robot_servo 툴을 호출해라.

* 
* 안전 제약 조건: 각 관절의 구동 범위를 절대로 넘지 마라.
* 특이점 회피: 일직선으로 손을 뻗는 명령(예: '하늘 높이 손 들어')이 오면, 관절 3번(팔꿈치)이나 5번(손목)이 완벽한 일직선(0도 주변)이 되지 않도록 약 5~10도 정도 굽혀진 안전한 각도(예: 0.15 rad 이상 오프셋)를 계산해서 주입해라."
* 

실제 구동 테스트 (명령어 예시):

* 
* 유저: "로봇 팔이 하늘을 향해 손을 번쩍 드는 자세를 취해줘. 특이점 조심하고!"
* LLM 행동: 특이점이 발생하지 않는 직립 유사 각도 [0.0, -1.57, 1.4, 0.0, 1.57, 0.0] 등을 계산하여 자동으로 move_robot_servo 툴을 호출 ➡️ MCP 서버가 이를 받아 8ms 주기로 잘게 쪼개 로봇에 servoJ 패킷 송신.
* 

------------------------------
## ⚠️ 개발 시 중요 주의사항 (체크리스트)

   1. 실시간성(Real-time) 보장: LLM 코드가 작동하는 메인 루프와 로봇에게 servoJ를 스트리밍하는 루프는 철저히 분리되어야 합니다. 위 예시처럼 asyncio 기반 비동기로 처리하거나, FastAPI 등을 쓸 경우 궤적 생성 전용 백그라운드 스레드(Thread)를 파서 로봇 통신 주기가 끊기지 않게 하세요. (주기가 끊기면 로봇이 무섭게 덜컥거립니다.)
   2. 시뮬레이터 우선 검증: 실제 로봇에 연결하기 전, URSim 같은 가상 로봇 시뮬레이터를 켜고 IP를 가상 IP로 잡아 LLM이 이상한 각도를 주입해도 안전한지 먼저 눈으로 확인하세요.
   3. Emergency Stop (비상 정지): 만약의 사태를 대비해 언제든 즉시 제어를 끊고 rtde_c.servoStop() 또는 rtde_c.stopJ()를 호출할 수 있는 비상 정지(E-Stop)용 Tool도 MCP 서버에 함께 구현해 두는 것을 강력히 권장합니다.

실제 구축하시려는 협동로봇의 정확한 제조사/모델명이나, 현재 생각 중인 LLM 프레임워크(LangChain, Claude API 등)를 알려주시면 통신 라이브러리 및 클라이언트 코드를 더 맞춤형으로 좁혀서 짜드릴 수 있습니다. 어떻게 진행해 볼까요?

[1] [https://devocean.sk.com](https://devocean.sk.com/blog/techBoardDetail.do?ID=167498&boardType=techBlog)
[2] [https://apidog.com](https://apidog.com/kr/blog/director-mcp-server/)
[3] [https://apidog.com](https://apidog.com/kr/blog/springboot-mcp-server-guide/)
[4] [https://insight.infograb.net](https://insight.infograb.net/blog/2025/01/22/mcp/)
