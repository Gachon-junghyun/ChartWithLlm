"""
03_llm_pipeline.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Gemini 2-pass inference pipeline (Google GenAI)

Flow
────
  Read pairs.jsonl line by line
      │
      ▼
  [Pass 1] Show X chart only
    → LLM: Pattern analysis + hypothesis generation
    → "This chart shows a pullback in uptrend. Expected +5~8% in 10 days. Basis: ..."
      │
      ▼
  [Pass 2] Show X chart + Y actual results together
    → LLM: Hypothesis verification / revision
    → "Actual result: -3%. Hypothesis failed. Reason: breakout attempt without volume, ..."
      │
      ▼
  Save to llm_output/annotated.jsonl
    {original pair data + hypothesis + verdict + reason + outcome_label}

Config
────
  GEMINI_API_KEY : env var (export GEMINI_API_KEY=...)
  MODEL          : Gemini model name
  BATCH_SAVE     : How many records to process before saving
"""

import argparse
import json
import os
import sys
import time
import logging
from pathlib import Path

from google import genai
from google.genai import types

# ── 경로 ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
XY_PATH  = BASE_DIR / "xy_pairs" / "pairs.jsonl"
OUT_DIR  = BASE_DIR / "llm_output"
OUT_DIR.mkdir(exist_ok=True)
OUT_PATH = OUT_DIR / "annotated.jsonl"

# ── Gemini 설정 ────────────────────────────────────────────────────────────────
MODEL      = "gemini-2.5-flash"
BATCH_SAVE = 10                  # 10건마다 저장

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ── Pattern taxonomy (fixed) ──────────────────────────────────────────────────
# 패턴명은 가격 구조(모양)만 기술. 미래 방향 정보 포함 금지.
# "Breakout / Bounce / Correction" 같은 방향 암시 단어 제거 → LLM이 이름이 아닌
# 실제 차트를 읽어서 방향을 독립적으로 판단하게 강제.

PATTERN_TAXONOMY = [
    "Trend Pullback Zone",              # 상승추세 중 단기 조정 구간 (재개 여부 미정)
    "Range Consolidation Upper Test",   # 박스권 상단 인근 테스트 (돌파/실패 미정)
    "Range Consolidation Lower Test",   # 박스권 하단 인근 테스트 (이탈/지지 미정)
    "Ascending Triangle",               # 수렴 삼각형 — 저점 상승 + 고점 수평
    "Descending Triangle",              # 수렴 삼각형 — 고점 하락 + 저점 수평
    "Head and Shoulders",               # 머리어깨형 천장 구조
    "Inverse Head and Shoulders",       # 역머리어깨형 바닥 구조
    "Double Bottom",                    # 이중 바닥 (W형 구조)
    "Double Top",                       # 이중 천장 (M형 구조)
    "MA Bullish Cross",                 # MA20이 MA60 위로 교차한 상태
    "MA Bearish Cross",                 # MA20이 MA60 아래로 교차한 상태
    "Post Sharp Decline Zone",          # 단기 급락 직후 구간 (반등/지속 미정)
    "Post Sharp Rally Zone",            # 단기 급등 직후 구간 (조정/지속 미정)
    "Rising Channel",                   # 고점·저점 모두 우상향 채널
    "Falling Channel",                  # 고점·저점 모두 우하향 채널
    "Support Level Test",               # 수평 지지선 인근 (지지/이탈 미정)
    "Resistance Level Test",            # 수평 저항선 인근 (돌파/실패 미정)
    "Bollinger Band Squeeze",           # 볼린저 밴드 수축 구간 (방향 미정)
    "RSI Overbought Zone",              # RSI 70 초과 상태 (조정/지속 미정)
    "RSI Oversold Zone",                # RSI 30 미만 상태 (반등/추가하락 미정)
    "Other",                            # 위 구조에 해당하지 않음
]

TAXONOMY_STR = "\n".join(f"  - {p}" for p in PATTERN_TAXONOMY)

# ── 프롬프트 템플릿 ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert in Korean stock chart pattern analysis.
Your role is to extract technical features from three synchronized ASCII sub-charts
and use those features to predict the next 10 trading days' price direction.

═══════════════════════════════════════════════════════
HOW TO READ EACH CHART SECTION
═══════════════════════════════════════════════════════

▌ CHART 1 — Candlestick Price Chart (top section)
  Left = past, Right = most recent (today)
  █  Bullish candle body (close ≥ open)
  ░  Bearish candle body (close < open)
  ─  Doji (open ≈ close, indecision)
  │  Wick (high/low range beyond the body)
     (empty space = no price at that row)
  .  MA20 line  |  -  MA60 line
  ^  Bollinger upper band  |  ·  Bollinger middle  |  v  Bollinger lower band

▌ CHART 2 — RSI(14) Sub-chart (middle section)
  Fixed scale 0–100.  Reference lines: 70 (─), 50 (:), 30 (─)
  *  RSI dot (neutral zone, 30–70)
  +  RSI > 70 (overbought — caution for longs)
  =  RSI < 30 (oversold — watch for reversal)
  Reading tip: track RSI position vs. 70/50/30 lines and its recent trajectory (left→right).

▌ CHART 3 — Rolling OBV Sub-chart (bottom section)
  Rolling sum of signed volume over the last 20 trading days.
  OBV_roll[t] = Σ sign(ΔClose) × Volume for the 20 days ending at t
  Normalized to the visible range (min–max within the 2-month window).
  █  Net buying pressure (positive rolling OBV)
  ░  Net selling pressure (negative rolling OBV)
  ─  Neutral zero line
  Reading tip: rising bars (█ growing taller) = accumulation; shrinking or flipping to ░ = distribution.

═══════════════════════════════════════════════════════
FEATURE EXTRACTION RULES
═══════════════════════════════════════════════════════
Do NOT assume features — read them directly from the grids:
1. Price structure: describe the trajectory of candle bodies left→right.
2. MA relationship: is price above/below MA20? Is MA20 above/below MA60?
3. Bollinger state: are bands widening (^/v far apart) or squeezing (close together)?
4. RSI trajectory: where is the dot line relative to 70/50/30? Rising or falling right side?
5. OBV momentum: are bars predominantly █ or ░? Growing or shrinking toward the right?
6. Volume: are vol bars getting taller or shorter recently?

═══════════════════════════════════════════════════════
DIRECTION PREDICTION RULES (MANDATORY)
═══════════════════════════════════════════════════════
- predicted_direction MUST be UP or DOWN. FLAT is NOT valid.
  When uncertain, pick the higher-probability side and set confidence LOW.
- Bullish signals: price above MA20+MA60 stack, RSI 45–65 rising, OBV bars █ growing,
  volume expanding on up candles, Bollinger middle acting as support.
- Bearish signals: price below MA stack, RSI > 70 falling or < 30 continuing down,
  OBV bars ░ growing, volume expanding on down candles, upper band resistance.
- Bollinger squeeze (bands tight): direction from MA stack + OBV side.
- Conflicting signals: count aligned vs. misaligned signals; lower confidence accordingly.

CONFIDENCE:
- HIGH   : 4+ signals clearly aligned across all three chart layers
- MEDIUM : 2–3 signals aligned, others mixed
- LOW    : signals contradictory or insufficient

All output must be in English."""


def pass1_prompt(pair: dict) -> str:
    """Pass 1: Multi-chart feature extraction → structure identification → direction prediction.

    3-step flow:
    1. Read all three chart layers and extract features
    2. Identify price structure pattern
    3. Predict direction based only on extracted features (not pattern name)
    """
    return f"""## Stock: {pair['name']} ({pair['ticker']})
## Period: {pair['x_start']} ~ {pair['x_end']} (2 months, ~42 trading days)

### Combined Chart (Price │ Volume │ RSI │ OBV)
{pair['x_chart']}

---
Follow the three steps below in order.

**STEP 1 — Extract features from ALL THREE chart layers**
Do NOT make assumptions — read directly from the grids above.

From the PRICE chart:
- Price trajectory: Is the rightmost price level higher, lower, or flat vs. the left?
- Recent candles (rightmost ~10 columns): Count bull (█) vs. bear (░) candles.
- MA position: Is price above or below the . (MA20) line? Above or below - (MA60)?
- MA alignment: Is MA20 (.) above or below MA60 (-)? Are they converging or diverging?
- Bollinger bands: Are ^ and v far apart (expanding) or close (squeezing)?
  Is price near ^ (upper), v (lower), or · (middle)?
- Volume bars: Are bars getting taller or shorter in the rightmost ~10 columns?

From the RSI chart:
- Where is the * / ^ / v dot line relative to the ─ reference lines (70 and 30)?
- Is the RSI dot line rising (moving up) or falling toward the right?
- Any ^ (>70 overbought) or v (<30 oversold) chars visible?

From the OBV chart:
- Are bars predominantly █ (buying) or ░ (selling)?
- Is the bar height growing or shrinking toward the right?
- Any clear flip from █ to ░ or vice versa?

**STEP 2 — Identify the price STRUCTURE**
Select exactly one pattern name from the taxonomy below.
Base your selection on the PRICE chart shape you observed in Step 1.
The pattern name describes structure — NOT future direction.

{TAXONOMY_STR}

**STEP 3 — Predict direction from Step 1 features ONLY**
Use the features extracted in Step 1 as evidence.
The pattern name from Step 2 must NOT influence your direction call.

CONSTRAINTS:
- predicted_direction MUST be UP or DOWN. Never output FLAT.
- If uncertain, pick the higher-probability direction and set confidence to LOW.
- HIGH confidence requires 4+ signals (across price, RSI, OBV, volume) all aligned.
- Cite specific chart observations: e.g. "RSI dot above 70 line", "OBV bars █ growing last 5 cols".

[Output — JSON only, no extra text]
{{
  "chart_features": {{
    "price_trajectory": "describe left→right movement",
    "recent_bull_bear_ratio": "e.g. 7 bull / 3 bear in last 10 candles",
    "ma_position": "e.g. price above MA20 (.), MA20 above MA60 (-) — bullish stack",
    "bollinger_state": "e.g. bands squeezing, price at middle band",
    "volume_trend": "e.g. vol bars shrinking last 10 cols — declining interest",
    "rsi_reading": "e.g. RSI dot (*) near 60, rising, no + or = chars visible",
    "obv_reading": "e.g. █ bars dominant and growing — accumulation signal"
  }},
  "pattern": "One from the taxonomy above (exact string match)",
  "hypothesis": "One sentence prediction citing chart_features with specifics",
  "predicted_direction": "UP or DOWN",
  "predicted_return_pct": number,
  "confidence": "HIGH or MEDIUM or LOW",
  "key_signals": [
    "Signal 1 from chart — specific (e.g. OBV bars flipped to █ in last 5 cols)",
    "Signal 2",
    "Signal 3"
  ],
  "risk_factors": ["Risk 1 — specific chart evidence", "Risk 2"]
}}"""


def pass2_prompt(pair: dict, hypothesis: dict) -> str:
    """Pass 2: Show X + Y results together and request verification.

    Key goal: instead of narrative explanations of 'why it worked/failed',
    extract specific measurable conditions that were already visible in the X chart
    — knowing the actual outcome.
    → These become testable rules in 05_backtest.py.
    """
    y_daily_str = "\n".join(
        f"  Day {d['day']} ({d['date']}): "
        f"O{d['open']:,.0f} H{d['high']:,.0f} L{d['low']:,.0f} C{d['close']:,.0f} "
        f"({d['ret_pct']:+.2f}%)"
        for d in pair["y_daily"]
    )

    return f"""## Stock: {pair['name']} ({pair['ticker']})
## X Period: {pair['x_start']} ~ {pair['x_end']}

### Your Previous Hypothesis (from chart reading)
- Pattern: {hypothesis.get('pattern', '?')}
- Hypothesis: {hypothesis.get('hypothesis', '?')}
- Predicted Direction: {hypothesis.get('predicted_direction', '?')} / Predicted Return: {hypothesis.get('predicted_return_pct', '?')}%
- Confidence: {hypothesis.get('confidence', '?')}
- Key signals cited: {'; '.join(hypothesis.get('key_signals', []))}

---
### Actual Results (Y: {pair['y_start']} ~ {pair['y_end']})
{y_daily_str}

Final Return: {pair['y_return_pct']:+.3f}%
Max Upside: {pair['y_max_up']:+.3f}% / Max Drawdown: {pair['y_max_down']:+.3f}%
Actual Direction: {pair['y_direction']}

---
You now know the actual outcome. Analyze the following.

**IMPORTANT**: Do NOT write a narrative.
Knowing the result, look back at your chart_features from Pass 1 and identify
**which specific chart-readable conditions (RSI position, OBV direction, MA stack, volume trend,
Bollinger state) were the decisive predictors**.
Conditions must be expressed as thresholds readable from the ASCII chart grids — NOT narrative.

[Output format — respond with JSON only]
{{
  "verdict": "SUCCESS or PARTIAL or FAILURE",
  "direction_correct": true or false,
  "verdict_reason": "One sentence — no narrative, numbers only",

  "measurable_success_condition": {{
    "description": "The decisive chart-readable condition that led to {pair['y_direction']}",
    "volume_signal": "Vol bar pattern (e.g. bars growing on bull candles, last 5 cols / N/A)",
    "rsi_signal": "RSI grid reading (e.g. dot above 50 line and rising / N/A)",
    "ma_signal": "MA grid reading (e.g. price dots above both . and - lines / N/A)",
    "bollinger_signal": "BB grid reading (e.g. price at · mid after squeeze / N/A)",
    "obv_signal": "OBV grid reading (e.g. █ bars growing last 8 cols / N/A)"
  }},

  "measurable_failure_condition": {{
    "description": "Chart-readable warning signs that would suggest opposite direction",
    "volume_signal": "Vol bar warning (e.g. shrinking bars despite price rise / N/A)",
    "rsi_signal": "RSI warning (e.g. ^ chars at top, dot falling rightward / N/A)",
    "ma_signal": "MA warning (e.g. price candles below . line + . below - / N/A)",
    "bollinger_signal": "BB warning (e.g. price at ^ upper after expand / N/A)",
    "obv_signal": "OBV warning (e.g. ░ bars appeared and growing / N/A)"
  }},

  "testable_rule": "IF [grid-readable condition A] AND [grid-readable condition B] THEN [direction] ELSE [opposite]"
}}"""


# ── Gemini 호출 ────────────────────────────────────────────────────────────────

def call_gemini(messages: list[dict], temperature: float = 0.3) -> str:
    """Google GenAI chat API 호출 → 응답 텍스트 반환"""
    client = _get_client()

    # system 메시지 분리
    system_msg = next((m["content"] for m in messages if m["role"] == "system"), None)

    # user / assistant → Gemini Contents 변환 (assistant → "model")
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


def sanitize_hypothesis(h: dict) -> dict:
    """Pass 1 결과 후처리 — FLAT 예측 차단 및 confidence 검증"""
    # FLAT 방어: 모델이 지시를 무시하고 FLAT을 출력한 경우
    if h.get("predicted_direction") == "FLAT":
        log.warning("  [sanitize] FLAT 예측 감지 → predicted_return_pct 기반으로 방향 보정")
        ret = h.get("predicted_return_pct", 0)
        h["predicted_direction"] = "UP" if ret >= 0 else "DOWN"
        # FLAT을 출력했다는 건 확신이 없다는 뜻이므로 confidence 강등
        if h.get("confidence") == "HIGH":
            h["confidence"] = "MEDIUM"
        h["_flat_corrected"] = True  # 보정 이력 기록

    # predicted_return_pct 부호와 방향 불일치 경고
    ret = h.get("predicted_return_pct", 0)
    direction = h.get("predicted_direction", "")
    if direction == "UP" and ret < 0:
        log.warning(f"  [sanitize] 방향/수익률 불일치: UP but {ret}% → 수익률 부호 반전")
        h["predicted_return_pct"] = abs(ret)
    elif direction == "DOWN" and ret > 0:
        log.warning(f"  [sanitize] 방향/수익률 불일치: DOWN but +{ret}% → 수익률 부호 반전")
        h["predicted_return_pct"] = -abs(ret)

    return h


# ── 메인 ──────────────────────────────────────────────────────────────────────

def check_gemini() -> bool:
    """Gemini API 연결 확인"""
    try:
        client = _get_client()
        # 최소 호출로 키 유효성 확인
        r = client.models.generate_content(
            model=MODEL,
            contents="hi",
            config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=4),
        )
        log.info(f"Gemini API 연결 성공 — 모델: {MODEL}")
        return True
    except Exception as e:
        log.error(f"Gemini API 연결 실패: {e}")
        return False


def get_processed_ids() -> set:
    """Get list of already processed IDs (prevents duplicates on restart)"""
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
    parser = argparse.ArgumentParser(description="Gemini 2-pass inference pipeline")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="처리할 최대 샘플 수 (예: --limit 10 → 첫 1배치만 실행 후 종료)"
    )
    args = parser.parse_args()

    if not XY_PATH.exists():
        log.error(f"pairs.jsonl not found. Run 02_generate_xy.py first.")
        return

    if not check_gemini():
        return

    processed = get_processed_ids()
    log.info(f"Already processed: {len(processed)} samples")

    with open(XY_PATH, "r", encoding="utf-8") as f:
        pairs = [json.loads(l) for l in f if l.strip()]

    # --limit 적용: 미처리 샘플 중 랜덤 N개 추출 (종목 다양성 확보)
    if args.limit is not None:
        import random
        unprocessed = [p for p in pairs if p["id"] not in processed]
        random.shuffle(unprocessed)
        pairs = unprocessed[: args.limit]
        tickers = set(p["ticker"] for p in pairs)
        log.info(f"--limit {args.limit} 적용: {len(pairs)}개 랜덤 샘플 ({len(tickers)}개 종목)")

    total   = len(pairs)
    done    = 0
    skipped = 0
    errors  = 0
    buffer  = []

    log.info(f"Total pairs to process: {total}")

    for i, pair in enumerate(pairs, 1):
        if pair["id"] in processed:
            skipped += 1
            continue

        log.info(f"[{i}/{total}] {pair['ticker']} {pair['name']} | {pair['x_end']}")

        try:
            # ── Pass 1: Generate hypothesis ────────────────────────────────
            messages_p1 = [
                {"role": "system",  "content": SYSTEM_PROMPT},
                {"role": "user",    "content": pass1_prompt(pair)},
            ]
            raw_p1     = call_gemini(messages_p1)
            hypothesis = sanitize_hypothesis(extract_json(raw_p1))
            log.info(f"  P1: {hypothesis.get('predicted_direction','?')} "
                     f"{hypothesis.get('predicted_return_pct','?')}% | "
                     f"{hypothesis.get('confidence','?')}"
                     + (" [FLAT→보정]" if hypothesis.get("_flat_corrected") else ""))

            # ── Pass 2: Verify & extract lessons ───────────────────────────
            messages_p2 = messages_p1 + [
                {"role": "assistant", "content": raw_p1},
                {"role": "user",      "content": pass2_prompt(pair, hypothesis)},
            ]
            raw_p2  = call_gemini(messages_p2)
            verdict = extract_json(raw_p2)
            log.info(f"  P2: {verdict.get('verdict','?')} — "
                     f"{verdict.get('verdict_reason','')[:60]}...")

            # ── Merge results ──────────────────────────────────────────────
            result = {
                **pair,
                "hypothesis":         hypothesis,
                "verdict":            verdict,
                "processing_time_ms": int(time.time() * 1000),
            }
            buffer.append(result)
            done += 1

        except Exception as e:
            log.error(f"  Error: {e}")
            errors += 1
            continue

        # Batch save
        if len(buffer) >= BATCH_SAVE:
            with open(OUT_PATH, "a", encoding="utf-8") as f:
                for r in buffer:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            log.info(f"  Checkpoint saved: {len(buffer)} records")
            buffer.clear()

    # Save remaining
    if buffer:
        with open(OUT_PATH, "a", encoding="utf-8") as f:
            for r in buffer:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    log.info(f"\nDone: processed {done} / skipped {skipped} / errors {errors}")
    log.info(f"Output: {OUT_PATH}")


if __name__ == "__main__":
    main()
