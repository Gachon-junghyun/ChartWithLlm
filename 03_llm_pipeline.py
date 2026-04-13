"""
03_llm_pipeline.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Ollama Gemma4 2-pass 추론 파이프라인

흐름
────
  pairs.jsonl 한 줄씩 읽기
      │
      ▼
  [Pass 1] X 차트만 보여줌
    → LLM: 패턴 분석 + 가설 생성
    → "이 차트는 눌림목 이후 돌파 가능성. 10일 후 +5~8% 예상. 근거: ..."
      │
      ▼
  [Pass 2] X 차트 + Y 결과 함께 보여줌
    → LLM: 가설 검증 / 수정
    → "실제 결과는 -3%. 가설 실패. 실패 이유: 거래량 수반 없이 상단 저항 돌파 시도, ..."
      │
      ▼
  llm_output/annotated.jsonl 저장
    {원본 pair 데이터 + hypothesis + verdict + reason + outcome_label}

설정
────
  OLLAMA_URL  : Ollama API 엔드포인트
  MODEL       : 사용할 모델명
  BATCH_SAVE  : 몇 건마다 중간 저장할지
"""

import json
import sys
import time
import logging
from pathlib import Path

import requests

# ── 경로 ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
XY_PATH  = BASE_DIR / "xy_pairs" / "pairs.jsonl"
OUT_DIR  = BASE_DIR / "llm_output"
OUT_DIR.mkdir(exist_ok=True)
OUT_PATH = OUT_DIR / "annotated.jsonl"

# ── Ollama 설정 ────────────────────────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL      = "gemma3:27b"        # ollama pull gemma3:27b
BATCH_SAVE = 10                  # 10건마다 저장
TIMEOUT    = 120                 # 요청 타임아웃(초)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ── 패턴 분류체계 (고정 taxonomy) ─────────────────────────────────────────────
# LLM이 자유 생성 대신 이 목록에서 하나를 선택하도록 강제
# → 04_rule_extractor의 group_by_pattern()이 의미 있는 묶음을 만들 수 있음

PATTERN_TAXONOMY = [
    "눌림목",           # 상승 추세 중 단기 조정 후 재상승
    "박스권 돌파",       # 횡보 구간 상단 이탈
    "박스권 이탈",       # 횡보 구간 하단 이탈
    "삼각수렴 상향",     # 수렴 후 상향 돌파
    "삼각수렴 하향",     # 수렴 후 하향 이탈
    "헤드앤숄더",        # 천장 패턴
    "역헤드앤숄더",      # 바닥 패턴
    "이중바닥",          # W 패턴
    "이중천장",          # M 패턴
    "골든크로스",        # MA20이 MA60 상향 돌파
    "데드크로스",        # MA20이 MA60 하향 이탈
    "급락 후 반등",      # 단기 급락 후 매수세 유입
    "급등 후 조정",      # 단기 급등 후 차익 실현
    "상승채널",          # 고점·저점 동시 우상향
    "하락채널",          # 고점·저점 동시 우하향
    "지지선 반등",       # 수평 지지선에서 반등
    "저항선 돌파",       # 수평 저항선 상향 돌파
    "볼린저 수축 돌파",  # 밴드 수축 후 방향 돌파
    "RSI 과매수 조정",   # RSI 70 초과 후 되돌림
    "RSI 과매도 반등",   # RSI 30 미만 후 반등
    "기타",             # 위 패턴에 해당 없음
]

TAXONOMY_STR = "\n".join(f"  - {p}" for p in PATTERN_TAXONOMY)

# ── 프롬프트 템플릿 ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """당신은 한국 주식 차트 패턴 분석 전문가입니다.
텍스트 차트를 읽고 시장 패턴을 파악하여 미래 주가 방향을 예측하는 역할을 합니다.

캔들 문자 체계:
  █  양봉 몸통 (종가 ≥ 시가)
  ░  음봉 몸통 (종가 < 시가)
  ─  도지 (시가 ≈ 종가)
  │  꼬리 (고가/저가)
  0  빈칸
  .  MA20 / -  MA60 / ^v  볼린저밴드

차트는 왼쪽이 과거, 오른쪽이 현재입니다.
분석은 반드시 한국어로 작성하세요."""


def pass1_prompt(pair: dict) -> str:
    """Pass 1: X 차트만 보여주고 가설 생성 요청.
    패턴명은 PATTERN_TAXONOMY 중 하나를 선택하도록 강제 → 04에서 의미 있는 grouping 보장.
    """
    return f"""## 종목: {pair['name']} ({pair['ticker']})
## 기간: {pair['x_start']} ~ {pair['x_end']} (6개월)

### 텍스트 차트
{pair['x_chart']}

### 시장 상태 (메타데이터)
{pair['x_meta']}

---
위 차트를 분석하여 다음 10 거래일 예측을 작성하세요.

**패턴명은 반드시 아래 목록 중 하나만 사용하세요:**
{TAXONOMY_STR}

[출력 형식 — 반드시 아래 JSON으로만 응답]
{{
  "pattern": "위 목록 중 하나 (정확히 일치해야 함)",
  "hypothesis": "구체적인 가설 — 측정 가능한 수치 포함 (예: 거래량 전일比 1.5배 + 볼린저 상단 돌파 → 목표 +5~8%)",
  "predicted_direction": "UP 또는 DOWN 또는 FLAT",
  "predicted_return_pct": 숫자 (예: 5.0),
  "confidence": "HIGH 또는 MEDIUM 또는 LOW",
  "key_signals": [
    "신호1 — 수치 포함 (예: RSI=58, MA20 위 3% 위치)",
    "신호2",
    "신호3"
  ],
  "risk_factors": ["리스크1 — 수치 포함", "리스크2"]
}}"""


def pass2_prompt(pair: dict, hypothesis: dict) -> str:
    """Pass 2: X + Y 결과 보여주고 검증 요청.

    핵심 변경: '왜 됐냐/안 됐냐' 서사 대신
    '결과를 알고 있는 지금, X 차트에서 미리 읽을 수 있었던
    구체적·측정 가능한 조건'을 뽑아내도록 유도.
    → 이후 05_backtest.py에서 실제로 테스트 가능한 룰이 됨.
    """
    y_daily_str = "\n".join(
        f"  {d['day']}일차 ({d['date']}): "
        f"시{d['open']:,.0f} 고{d['high']:,.0f} 저{d['low']:,.0f} 종{d['close']:,.0f} "
        f"({d['ret_pct']:+.2f}%)"
        for d in pair["y_daily"]
    )

    return f"""## 종목: {pair['name']} ({pair['ticker']})
## X 기간: {pair['x_start']} ~ {pair['x_end']}

### 당신의 이전 가설
- 패턴: {hypothesis.get('pattern', '?')}
- 가설: {hypothesis.get('hypothesis', '?')}
- 예측 방향: {hypothesis.get('predicted_direction', '?')} / 예측 수익률: {hypothesis.get('predicted_return_pct', '?')}%
- 신뢰도: {hypothesis.get('confidence', '?')}

---
### 실제 결과 (Y: {pair['y_start']} ~ {pair['y_end']})
{y_daily_str}

최종 수익률: {pair['y_return_pct']:+.3f}%
최대 상승: {pair['y_max_up']:+.3f}% / 최대 낙폭: {pair['y_max_down']:+.3f}%
결과 방향: {pair['y_direction']}

---
결과를 확인했습니다. 이제 다음을 분석하세요.

**중요**: "왜 이렇게 됐는지" 서사를 쓰는 것이 아닙니다.
결과를 이미 알고 있는 상태에서, **X 차트의 메타데이터(RSI·거래량·MA 위치·볼린저)에서
미리 읽을 수 있었던 측정 가능한 수치 조건**을 찾아내세요.
이 조건은 나중에 새 차트에 그대로 적용해서 테스트할 수 있어야 합니다.

[출력 형식 — 반드시 아래 JSON으로만 응답]
{{
  "verdict": "SUCCESS 또는 PARTIAL 또는 FAILURE",
  "direction_correct": true 또는 false,
  "verdict_reason": "핵심 이유 1문장 (서사 금지 — 수치 기반으로)",

  "measurable_success_condition": {{
    "description": "이 패턴이 {pair['y_direction']}으로 간 결정적 조건 (X에서 이미 보였던 것)",
    "volume_signal": "거래량 조건 (예: 전일比 1.5배 이상 / 해당없음)",
    "rsi_signal": "RSI 조건 (예: 50~65 사이 / 해당없음)",
    "ma_signal": "MA 위치 조건 (예: 종가가 MA20 위 + MA20 우상향 / 해당없음)",
    "bollinger_signal": "볼린저 조건 (예: 중단 돌파 후 상단 향해 확장 / 해당없음)"
  }},

  "measurable_failure_condition": {{
    "description": "이 패턴이 반대 방향으로 갔을 때의 조건 (X에서 경고 신호)",
    "volume_signal": "거래량 경고 (예: 거래량 감소 동반 / 해당없음)",
    "rsi_signal": "RSI 경고 (예: RSI > 70 과매수 / 해당없음)",
    "ma_signal": "MA 경고 (예: MA20 아래 + 역배열 / 해당없음)",
    "bollinger_signal": "볼린저 경고 (예: 상단 밴드 터치 후 수축 / 해당없음)"
  }},

  "testable_rule": "IF [조건A 수치] AND [조건B 수치] THEN [방향] ELSE [반대방향]"
}}"""


# ── Ollama 호출 ────────────────────────────────────────────────────────────────

def call_ollama(messages: list[dict]) -> str:
    """Ollama chat API 호출 → 응답 텍스트 반환"""
    payload = {
        "model":    MODEL,
        "messages": messages,
        "stream":   False,
        "options": {
            "temperature": 0.3,
            "top_p":       0.9,
        }
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


def extract_json(text: str) -> dict:
    """응답에서 JSON 블록 추출"""
    # ```json ... ``` 형태 처리
    if "```" in text:
        start = text.find("```")
        end   = text.rfind("```")
        text  = text[start:end].replace("```json", "").replace("```", "").strip()
    # { ... } 직접 추출
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start >= 0 and end > start:
        text = text[start:end]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw_response": text}


# ── 메인 ──────────────────────────────────────────────────────────────────────

def check_ollama() -> bool:
    """Ollama 서버 연결 확인"""
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        log.info(f"Ollama 연결 OK. 사용 가능 모델: {models}")
        if not any(MODEL.split(":")[0] in m for m in models):
            log.warning(f"'{MODEL}' 모델 없음. `ollama pull {MODEL}` 먼저 실행하세요.")
            return False
        return True
    except Exception as e:
        log.error(f"Ollama 연결 실패: {e}")
        log.error("Ollama가 실행 중인지 확인하세요: `ollama serve`")
        return False


def get_processed_ids() -> set:
    """이미 처리된 ID 목록 (재시작 시 중복 방지)"""
    if not OUT_PATH.exists():
        return set()
    ids = set()
    with open(OUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                ids.add(json.loads(line)["id"])
            except Exception:
                pass
    return ids


def main():
    if not XY_PATH.exists():
        log.error(f"pairs.jsonl 없음. 먼저 02_generate_xy.py 실행하세요.")
        return

    if not check_ollama():
        return

    processed = get_processed_ids()
    log.info(f"이미 처리된 샘플: {len(processed)}개")

    with open(XY_PATH, "r", encoding="utf-8") as f:
        pairs = [json.loads(l) for l in f if l.strip()]

    total   = len(pairs)
    done    = 0
    skipped = 0
    errors  = 0
    buffer  = []

    log.info(f"처리 대상: {total}개 페어")

    for i, pair in enumerate(pairs, 1):
        if pair["id"] in processed:
            skipped += 1
            continue

        log.info(f"[{i}/{total}] {pair['ticker']} {pair['name']} | {pair['x_end']}")

        try:
            # ── Pass 1: 가설 생성 ──────────────────────────────────────────
            messages_p1 = [
                {"role": "system",  "content": SYSTEM_PROMPT},
                {"role": "user",    "content": pass1_prompt(pair)},
            ]
            raw_p1     = call_ollama(messages_p1)
            hypothesis = extract_json(raw_p1)
            log.info(f"  P1: {hypothesis.get('predicted_direction','?')} "
                     f"{hypothesis.get('predicted_return_pct','?')}% | "
                     f"{hypothesis.get('confidence','?')}")

            # ── Pass 2: 검증 & 교훈 ───────────────────────────────────────
            messages_p2 = messages_p1 + [
                {"role": "assistant", "content": raw_p1},
                {"role": "user",      "content": pass2_prompt(pair, hypothesis)},
            ]
            raw_p2  = call_ollama(messages_p2)
            verdict = extract_json(raw_p2)
            log.info(f"  P2: {verdict.get('verdict','?')} — "
                     f"{verdict.get('lesson','')[:60]}...")

            # ── 결과 합치기 ────────────────────────────────────────────────
            result = {
                **pair,
                "hypothesis":         hypothesis,
                "verdict":            verdict,
                "processing_time_ms": int(time.time() * 1000),
            }
            buffer.append(result)
            done += 1

        except Exception as e:
            log.error(f"  오류: {e}")
            errors += 1
            continue

        # 배치 저장
        if len(buffer) >= BATCH_SAVE:
            with open(OUT_PATH, "a", encoding="utf-8") as f:
                for r in buffer:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            log.info(f"  중간 저장: {len(buffer)}건")
            buffer.clear()

    # 나머지 저장
    if buffer:
        with open(OUT_PATH, "a", encoding="utf-8") as f:
            for r in buffer:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    log.info(f"\n완료: 처리 {done} / 스킵 {skipped} / 오류 {errors}")
    log.info(f"결과: {OUT_PATH}")


if __name__ == "__main__":
    main()
