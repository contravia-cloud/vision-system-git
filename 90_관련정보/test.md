---

## 🛠️ 구현 가이드

### 1. 전제 조건 (Prerequisites)
로봇과의 RTDE 통신 및 MCP 서버 구성을 위해 아래 라이브러리가 필요합니다.
```bash
pip install mcp url_rtde asyncio
```

### 2. MCP 서버 소스 코드 (`mcp_server.py`)
이 서버는 LLM에게 `move_robot_servo` 툴을 노출하고, 명령을 받으면 목표 각도까지 선형 보간(Linear Interpolation)을 통해 실시간으로 `servoJ` 패킷을 스트리밍합니다.

```python
import asyncio
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.types as types
import rtde_control

# 1. MCP 서버 및 로봇 컨트롤러 설정
server = Server("robot-servo-controller")
ROBOT_IP = "192.168.1.100"  # 실제 로봇 또는 URSim 시뮬레이터 IP 입력
rtde_c = None

try:
    rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
    print("✅ 로봇 RTDE 컨트롤러 연결 성공")
except Exception as e:
    print(f"❌ 로봇 연결 실패(시뮬레이션 모드 또는 IP 확인 필요): {e}")

# 2. LLM에게 노출할 Tool 정의
@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
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

# 3. Tool 실행 로직 (servoJ 실시간 제어 루프)
@server.call_tool()
async def handle_call_tool(name: str, arguments: dict | None) -> list[types.TextContent]:
    if name != "move_robot_servo":
        raise ValueError(f"Unknown tool: {name}")
        
    if not arguments or "target_joints" not in arguments:
        return [types.TextContent(type="text", text="오류: target_joints 인자가 필요합니다.")]

    target_q = arguments["target_joints"]
    duration = arguments.get("duration", 3.0)

    if rtde_c is None:
        return [types.TextContent(type="text", text=f"⚠️ 로봇 미연결 상태 (요청 각도: {target_q})")]

    try:
        # 현재 관절 위치 및 제어 주기 설정
        start_q = rtde_c.getActualQ()
        dt = 0.008  # 125Hz 주기 (e-Series의 경우 0.002초/500Hz 설정 권장)
        steps = int(duration / dt)
        
        # 실시간 스트리밍 루프
        for i in range(steps):
            # 목표 각도까지 선형 보간 계산
            alpha = i / steps
            current_target_q = [start_q[j] + alpha * (target_q[j] - start_q[j]) for j in range(6)]
            
            # servoJ(q, 속도, 가속도, 시간t, 룩어헤드시간, 게인)
            rtde_c.servoJ(current_target_q, 0.0, 0.0, dt, 0.1, 300)
            
            # 정밀 주기 동기화를 위한 비동기 슬립
            await asyncio.sleep(dt)
            
        rtde_c.servoStop()
        return [types.TextContent(type="text", text=f"✅ 성공: 목표 각도 {target_q}로 이동을 완료했습니다.")]

    except Exception as e:
        if rtde_c:
            rtde_c.servoStop()
        return [types.TextContent(type="text", text=f"❌ 로봇 제어 중 오류 발생: {str(e)}")]

# 4. 서버 메인 실행부 (Standard I/O 통신)
async def main():
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
```

---

## 🚀 실행 및 연동 방법

### Claude Desktop 클라이언트 연동
Claude Desktop 앱 설정 파일(`claude_desktop_config.json`)에 제작한 MCP 서버 스크립트를 아래와 같이 등록합니다.

* **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
* **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
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
```
*설정 파일 저장 후 Claude Desktop 앱을 완전히 종료했다가 다시 실행하면 우측 하단에 망치 아이콘(Tools)이 활성화됩니다.*

---

## 🤖 LLM 프롬프트 가이드 (System Prompt)

LLM이 로봇 제어 툴을 안전하게 사용할 수 있도록 아래 프롬프트를 **시스템 가이드라인**으로 주입하는 것을 강력히 권장합니다.

```text
너는 6자유도 협동로봇을 안전하게 제어하는 전문 AI 엔지니어다.
사용자가 자연어로 명령을 내리면 상황에 가장 적합한 각도(라디안 단위)를 계산해 `move_robot_servo` 툴을 호출해라.

[안전 제약 조건]
1. 로봇 각 관절의 물리적 구동 한계 범위를 절대로 초과하는 각도를 주입하지 마라.
2. 직립 특이점(Singularity) 회피: '하늘 높이 손 들어'와 같이 팔을 곧게 뻗는 명령이 올 경우, 
   관절 3번(팔꿈치)이나 관절 5번(손목)이 완벽한 일직선(0 rad 근처)이 되지 않도록 
   최소 5도~10도(약 0.1 ~ 0.17 rad) 가량 굽혀진 안전 오프셋 각도를 계산하여 툴을 호출해라.
```

---

## ⚠️ 주의사항 (Safety Notice)
1. **실시간성 차단 주의**: 코드 내에 대기 시간이 긴 블로킹(Blocking) 함수를 넣으면 `servoJ` 통신 주기가 깨져 로봇이 급격하게 요동칠 수 있습니다. 반드시 비동기(`asyncio`) 구조를 유지하세요.
2. **시뮬레이터 검증**: 실제 하드웨어 로봇에 연결하기 전, **URSim 가상 시뮬레이터** 환경에서 LLM의 각도 연산 결과를 충분히 모니터링 및 테스트한 뒤 적용하십시오.
3. **비상 정지**: 만약의 오동작 상황을 대비해 물리적인 비상 정지 버튼(E-Stop) 위치를 상시 확보하고 작업하십시오.
