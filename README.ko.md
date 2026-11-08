# Hermes Runtime용 AGY Brain-Only Wrapper Server (`/v1/responses`)

[English](README.md)

이 프로젝트는 **Antigravity CLI (`agy`)** 를 **Hermes Agent Runtime**의 순수 추론 백엔드(두뇌)로 사용하는 **OpenAI Responses API (`POST /v1/responses`)** 호환 wrapper 서버다.

서버 자체는 외부 Python 패키지 없이 **Python 3 표준 라이브러리(`asyncio`)** 만으로 실행된다.

---

## 핵심 동작

### Brain과 Body의 역할 분리

- **Wrapper / AGY (Brain):** `--settings '{"permissions":{"allow":[],"deny":["*"]}}'` 및 `agy_runtime_cwd/GEMINI.md` 규칙으로 AGY 내장 도구 실행을 차단한다. AGY는 한 턴에서 추론 결과와 도구 호출 결정(`content` + `tool_calls`)만 반환한다.
- **Hermes Runtime (Body):** Wrapper가 반환한 `type: "function_call"` 항목을 받아 실제 도구(`terminal`, `clarify`, `memory`, `patch` 등)를 로컬에서 실행한다. 실행 결과는 `type: "function_call_output"`으로 다시 Wrapper에 전달한다.

### 지원 API 및 기능

- OpenAI Responses API: `POST /v1/responses`
  - `instructions`
  - 문자열 또는 `message`, `function_call`, `function_call_output` 배열 형태의 `input`
  - 인메모리 `previous_response_id` 상태 체이닝
  - `tools` 및 `tool_choice` (`auto`, `none`, `required`, 특정 함수 강제)
  - `stream: true` 요청에 대한 Typed SSE 이벤트 (`response.created`, `response.output_item.added`, `response.function_call_arguments.delta`, `response.completed` 등)
- OpenAI Chat Completions 호환 API: `POST /v1/chat/completions`
- 모델 목록: `GET /v1/models`
- 상태 확인: `GET /health`

---

## 디렉터리 구조

- `server.py`: 비동기 HTTP/1.1 + SSE 서버 및 API 엔드포인트
- `agy_runner.py`: 비대화형 brain-only 모드로 `agy` CLI를 실행하고 응답과 토큰 사용량을 추출
- `prompt_builder.py`: Responses API 요청, 대화 이력, 도구 스키마를 AGY 단일 프롬프트로 직렬화하고 JSON 응답을 파싱
- `schemas.py`: 내부 데이터 모델 (`dataclass`)
- `config.py`: 호스트, 포트, 기본 모델(`AGY_MODEL`), 추론 강도(`AGY_EFFORT`) 등의 환경변수 설정
- `agy_runtime_cwd/GEMINI.md`: AGY 서브프로세스의 격리 실행 폴더와 brain-only 지침
- `test_responses_api.py`: 일반 도구 호출, 멀티턴 `function_call_output`, `clarify` 도구의 SSE 스트리밍 검증 스크립트
- `test_server_safety.py`: 컨텍스트 및 요청 크기 제한 검증을 위한 pytest 테스트

---

## 실행

### 1. 서버 시작

`agy` CLI를 설치하고 인증한 후, 저장소 디렉터리에서 서버를 시작한다.

```bash
cd /path/to/agy_backend
python3 server.py
```

기본 주소는 `127.0.0.1:8000`이며, `~/.gemini/antigravity-cli/settings.json`에 설정된 기본 AGY 모델을 사용한다.

특정 모델과 추론 강도를 지정할 때는 다음과 같이 실행한다.

```bash
AGY_MODEL="gemini-4-argon-high" AGY_EFFORT="high" AGY_WRAPPER_PORT=8000 python3 server.py
```

### 2. Hermes 설정

`~/.hermes/config.yaml`에 다음과 같이 등록한다.

```yaml
model:
  default: "gemini-4-argon-high"   # 또는 "agy-hermes-backend", "gemini-3.8-flash-high" 등
  provider: "custom"
  base_url: "http://127.0.0.1:8000/v1"
  api_key: "agy-local"             # 로컬 wrapper용 더미 값
  api_mode: "codex_responses"
```

### 3. 리소스 제한 설정

기본값은 AGY 동시 실행 최대 2개, HTTP 요청 본문 최대 10 MB, HTTP 요청 읽기 제한시간 30초다.

```bash
MAX_CONCURRENT_AGY_PROCESSES=2 MAX_REQUEST_BODY_BYTES=10000000 \
HTTP_READ_TIMEOUT_SECONDS=30 AGY_MAX_PROMPT_CHARS=900000 python3 server.py
```

`/v1/responses`와 `/v1/chat/completions`는 같은 prompt 크기 상한을 적용한다. 초과 요청은 HTTP 413으로 반환한다.

---

## 검증

표준 라이브러리 기반의 통합 검증 스크립트는 다음과 같이 실행한다.

```bash
python3 test_responses_api.py
```

`test_server_safety.py`는 선택적으로 `pytest`가 설치된 환경에서 실행한다.

```bash
python3 -m pytest -q test_server_safety.py
```
