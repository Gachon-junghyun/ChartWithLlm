"""
04_rule_extractor.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
모순 케이스 탐지 → Ollama로 룰 추출 → 사전 MD 업데이트

흐름
────
  annotated.jsonl 읽기
      │
      ▼
  패턴별 그룹핑 (hypothesis.pattern 기준)
      │
      ▼
  같은 패턴인데 결과 반대인 쌍 탐지 (SUCCESS vs FAILURE)
      │
      ▼
  [LLM] "이 두 케이스의 차이점은? 언제 되고 언제 안 되는가?"
      │
      ▼
  rules/ 폴더에 패턴별 MD 저장
  rules/knowledge_base.md (통합 사전)
"""

import json
import os
import sys
import logging
from pathlib import Path
from itertools import combinations
from collections import defaultdict

from google import genai
from google.genai import types

# ── 경로 ──────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
ANNOTATED_PATH = BASE_DIR / "llm_output" / "annotated.jsonl"
RULES_DIR     = BASE_DIR / "rules"
RULES_DIR.mkdir(exist_ok=True)
KB_PATH       = RULES_DIR / "knowledge_base.md"

# ── Gemini 설정 ────────────────────────────────────────────────────────────────
MODEL = "gemini-2.5-flash"

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "API 키 없음. 환경변수 GEMINI_API_KEY 를 설정하세요.\n"
                "  export GEMINI_API_KEY=your_key_here"
            )
        _client = genai.Client(api_key=api_key)
    return _client

# 패턴 그룹당 최대 모순 쌍 수 (너무 많으면 처리 오래 걸림)
MAX_CONFLICT_PAIRS = 5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ── 유틸 ──────────────────────────────────────────────────────────────────────

def load_annotated() -> list[dict]:
    if not ANNOTATED_PATH.exists():
        log.error(f"annotated.jsonl 없음. 먼저 03_llm_pipeline.py 실행하세요.")
        return []
    with open(ANNOTATED_PATH, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def group_by_pattern(samples: list[dict]) -> dict[str, list[dict]]:
    """패턴명으로 샘플 그룹핑"""
    groups = defaultdict(list)
    for s in samples:
        pattern = s.get("hypothesis", {}).get("pattern", "UNKNOWN")
        groups[pattern].append(s)
    return dict(groups)


def find_conflict_pairs(group: list[dict]) -> list[tuple[dict, dict]]:
    """SUCCESS와 FAILURE 쌍 찾기"""
    successes = [s for s in group if s.get("verdict", {}).get("verdict") == "SUCCESS"]
    failures  = [s for s in group if s.get("verdict", {}).get("verdict") == "FAILURE"]

    pairs = []
    for s in successes:
        for f in failures:
            pairs.append((s, f))
            if len(pairs) >= MAX_CONFLICT_PAIRS:
                return pairs
    return pairs


def format_case(sample: dict, label: str) -> str:
    """단일 케이스를 LLM에 보낼 텍스트로 포맷.
    새 포맷: measurable_success/failure_condition + testable_rule 우선 사용.
    """
    h = sample.get("hypothesis", {})
    v = sample.get("verdict", {})

    # ── 측정 가능한 조건 (새 포맷) ───────────────────────────────────────────
    s_cond = v.get("measurable_success_condition", {})
    f_cond = v.get("measurable_failure_condition", {})

    def fmt_cond(cond: dict) -> str:
        if not cond:
            return "  (없음)"
        lines = []
        if cond.get("description"):
            lines.append(f"  설명: {cond['description']}")
        for key, label_k in [
            ("volume_signal",   "거래량"),
            ("rsi_signal",      "RSI"),
            ("ma_signal",       "MA"),
            ("bollinger_signal","볼린저"),
        ]:
            val = cond.get(key, "해당없음")
            if val and val != "해당없음":
                lines.append(f"  {label_k}: {val}")
        return "\n".join(lines) if lines else "  (없음)"

    testable = v.get("testable_rule", v.get("condition_success", "?"))

    return f"""=== {label}: {sample['ticker']} {sample['name']} ({sample['x_end']}) ===
패턴: {h.get('pattern', '?')}
가설: {h.get('hypothesis', '?')}
주요 신호: {', '.join(h.get('key_signals', []))}
리스크: {', '.join(h.get('risk_factors', []))}
메타데이터: {sample.get('x_meta', '')}

실제 결과: {sample['y_return_pct']:+.2f}% ({sample['y_direction']})
최대상승: {sample['y_max_up']:+.2f}% / 최대낙폭: {sample['y_max_down']:+.2f}%
판정: {v.get('verdict', '?')} | 방향 정확: {v.get('direction_correct', '?')}
판정 이유: {v.get('verdict_reason', '?')}

[성공 조건 — X에서 측정 가능한 수치]
{fmt_cond(s_cond)}

[실패 조건 — X에서 측정 가능한 수치]
{fmt_cond(f_cond)}

테스트 가능 룰: {testable}"""


def call_gemini(messages: list[dict], temperature: float = 0.2) -> str:
    """Google GenAI chat API 호출 → 응답 텍스트 반환"""
    client = _get_client()

    system_msg = next((m["content"] for m in messages if m["role"] == "system"), None)

    contents = []
    for m in messages:
        if m["role"] == "user":
            contents.append(
                types.Content(role="user", parts=[types.Part.from_text(text=m["content"])])
            )
        elif m["role"] == "assistant":
            contents.append(
                types.Content(role="model", parts=[types.Part.from_text(text=m["content"])])
            )

    config = types.GenerateContentConfig(
        temperature=temperature,
        top_p=0.9,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        **({"system_instruction": system_msg} if system_msg else {}),
    )

    response = client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=config,
    )
    return response.text.strip()


RULE_SYSTEM = """당신은 주식 차트 패턴 분석 룰을 도출하는 전문가입니다.
성공 케이스와 실패 케이스를 비교하여 명확한 조건 기반 룰을 한국어로 작성하세요.

핵심 원칙:
- 서사(이야기) 금지. 오직 측정 가능한 수치 조건만 작성하세요.
- 조건은 새 차트에 그대로 적용해서 테스트할 수 있어야 합니다.
- 예시: "거래량 전일比 1.5배 이상 + RSI 50~65 + MA20 위 → UP"
- 예시: "거래량 감소 + RSI > 70 → FAILURE 가능성 높음"
"""


def extract_rule_from_conflict(pattern: str, success: dict, failure: dict) -> str:
    """모순 쌍으로부터 측정 가능한 룰 추출"""
    prompt = f"""다음은 '{pattern}' 패턴에서 결과가 반대로 나온 두 케이스입니다.
각 케이스에는 X 차트에서 미리 관찰 가능했던 측정 조건이 포함되어 있습니다.

{format_case(success, '성공 케이스 (실제 결과: UP/SUCCESS)')}

{format_case(failure, '실패 케이스 (실제 결과: DOWN/FAILURE)')}

---
두 케이스의 측정 조건을 비교하여 다음 형식으로 룰을 작성하세요.
**반드시 수치 기반 조건만 사용하고, 서사적 설명은 쓰지 마세요.**

## {pattern} 패턴 룰

### 성공 조건 (X 차트에서 사전 관찰 가능한 수치)
- 거래량: (예: 전일比 N배 이상)
- RSI: (예: N~N 사이)
- MA 위치: (예: 종가가 MA20 위, MA20 우상향)
- 볼린저: (예: 중단 돌파 후 상단 향해 확장)

### 실패 조건 (X 차트에서 사전 관찰 가능한 경고 수치)
- 거래량: (예: 전일比 감소 또는 보합)
- RSI: (예: N 초과 과매수)
- MA 위치: (예: MA20 아래, 역배열)
- 볼린저: (예: 상단 터치 후 수축)

### 핵심 분기 조건
두 케이스를 가른 결정적 수치 차이 한 줄로:
"[측정조건A 수치]이면 → [방향] / [측정조건B 수치]이면 → [반대방향]"

### 적용 룰 (테스트 가능 형식)
IF [조건1] AND [조건2] THEN [예측방향] (신뢰도: HIGH/MEDIUM/LOW)
ELSE IF [조건3] THEN [반대방향]
"""
    messages = [
        {"role": "system", "content": RULE_SYSTEM},
        {"role": "user",   "content": prompt},
    ]
    return call_gemini(messages)


def synthesize_pattern_rules(pattern: str, all_rules: list[str]) -> str:
    """패턴의 여러 룰을 하나로 통합 — 수치 조건 중심"""
    if len(all_rules) == 1:
        return all_rules[0]

    rules_text = "\n\n---\n\n".join(
        f"[룰 {i+1}]\n{r}" for i, r in enumerate(all_rules)
    )
    prompt = f"""'{pattern}' 패턴에 대해 여러 케이스에서 도출된 룰들입니다.

{rules_text}

---
위 룰들을 통합하여 최종 통합 룰을 작성하세요.
**수치 조건만 남기고, 중복되거나 모순되는 조건은 더 많은 케이스를 지지하는 쪽으로 통합하세요.**
형식은 마크다운으로 작성하세요.

포함 필수 항목:
1. 공통 성공 수치 조건 (거래량 / RSI / MA / 볼린저 각각)
2. 공통 실패 수치 조건
3. 예외 케이스 (조건 충족했지만 실패한 경우)
4. 최종 IF-THEN 룰 (테스트 가능 형식)
"""

    messages = [
        {"role": "system", "content": RULE_SYSTEM},
        {"role": "user",   "content": prompt},
    ]
    return call_gemini(messages)


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    samples = load_annotated()
    if not samples:
        return

    log.info(f"총 {len(samples)}개 샘플 로드")

    # 통계
    verdicts = [s.get("verdict", {}).get("verdict", "UNKNOWN") for s in samples]
    log.info(f"SUCCESS: {verdicts.count('SUCCESS')} / "
             f"PARTIAL: {verdicts.count('PARTIAL')} / "
             f"FAILURE: {verdicts.count('FAILURE')}")

    groups = group_by_pattern(samples)
    log.info(f"패턴 종류: {len(groups)}개")
    for p, g in sorted(groups.items(), key=lambda x: -len(x[1])):
        log.info(f"  {p}: {len(g)}개")

    kb_sections = []   # 최종 지식베이스 섹션들

    for pattern, group in groups.items():
        log.info(f"\n패턴: '{pattern}' ({len(group)}개)")

        conflict_pairs = find_conflict_pairs(group)
        if not conflict_pairs:
            log.info(f"  모순 쌍 없음 (성공 또는 실패만 존재)")
            continue

        log.info(f"  모순 쌍: {len(conflict_pairs)}쌍")

        pattern_rules = []
        for j, (suc, fail) in enumerate(conflict_pairs, 1):
            log.info(f"  [{j}/{len(conflict_pairs)}] "
                     f"성공={suc['ticker']}({suc['x_end']}) vs "
                     f"실패={fail['ticker']}({fail['x_end']})")
            try:
                rule = extract_rule_from_conflict(pattern, suc, fail)
                pattern_rules.append(rule)
                log.info(f"  룰 추출 완료 ({len(rule)}자)")
            except Exception as e:
                log.error(f"  룰 추출 실패: {e}")

        if not pattern_rules:
            continue

        # 패턴별 룰 통합
        try:
            final_rule = synthesize_pattern_rules(pattern, pattern_rules)
        except Exception as e:
            log.error(f"통합 실패: {e}")
            final_rule = "\n\n---\n\n".join(pattern_rules)

        # 패턴별 개별 파일 저장
        safe_name = pattern.replace("/", "_").replace(" ", "_")
        pattern_path = RULES_DIR / f"{safe_name}.md"
        pattern_path.write_text(final_rule, encoding="utf-8")
        log.info(f"  저장: {pattern_path}")

        kb_sections.append(f"# {pattern}\n\n{final_rule}")

    # ── 통합 지식베이스 MD 생성 ────────────────────────────────────────────
    if kb_sections:
        kb_content = "# 차트 패턴 룰 사전 (Knowledge Base)\n\n"
        kb_content += f"생성일: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        kb_content += f"분석 샘플 수: {len(samples)}개\n\n"
        kb_content += "---\n\n"
        kb_content += "\n\n---\n\n".join(kb_sections)

        KB_PATH.write_text(kb_content, encoding="utf-8")
        log.info(f"\n통합 지식베이스 저장: {KB_PATH}")
    else:
        log.info("추출된 룰 없음")

    log.info("\n완료.")


if __name__ == "__main__":
    main()
