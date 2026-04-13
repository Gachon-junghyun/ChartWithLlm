"""
03_llm_pipeline.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Ollama Gemma4 2-pass inference pipeline

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
  OLLAMA_URL  : Ollama API endpoint
  MODEL       : Model name to use
  BATCH_SAVE  : How many records to process before saving
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
MODEL      = "gemma4:latest"     # ollama pull gemma4
BATCH_SAVE = 10                  # 10건마다 저장
TIMEOUT    = 120                 # 요청 타임아웃(초)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ── Pattern taxonomy (fixed) ──────────────────────────────────────────────────
# Force LLM to select from this list instead of free-form generation
# → Ensures meaningful grouping in 04_rule_extractor's group_by_pattern()

PATTERN_TAXONOMY = [
    "Pullback in Uptrend",          # Short-term correction during uptrend, then resumes
    "Range Breakout",               # Break above upper boundary of consolidation range
    "Range Breakdown",              # Break below lower boundary of consolidation range
    "Ascending Triangle Breakout",  # Breakout upward after triangular convergence
    "Descending Triangle Breakdown",# Breakdown downward after triangular convergence
    "Head and Shoulders",           # Topping pattern
    "Inverse Head and Shoulders",   # Bottoming pattern
    "Double Bottom",                # W pattern
    "Double Top",                   # M pattern
    "Golden Cross",                 # MA20 crosses above MA60
    "Dead Cross",                   # MA20 crosses below MA60
    "Bounce After Sharp Decline",   # Buying pressure after short-term sharp drop
    "Pullback After Sharp Rally",   # Profit-taking after short-term sharp rise
    "Rising Channel",               # Both highs and lows trending up
    "Falling Channel",              # Both highs and lows trending down
    "Support Level Bounce",         # Bounce from horizontal support level
    "Resistance Level Breakout",    # Break above horizontal resistance level
    "Bollinger Squeeze Breakout",   # Directional breakout after band contraction
    "RSI Overbought Correction",    # Pullback after RSI exceeds 70
    "RSI Oversold Bounce",          # Bounce after RSI drops below 30
    "Other",                        # Does not match any pattern above
]

TAXONOMY_STR = "\n".join(f"  - {p}" for p in PATTERN_TAXONOMY)

# ── 프롬프트 템플릿 ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert in Korean stock chart pattern analysis.
Your role is to read text-based charts, identify market patterns, and predict future price direction.

Candle character system:
  █  Bullish body (close >= open)
  ░  Bearish body (close < open)
  ─  Doji (open ≈ close)
  │  Wick (high/low)
  0  Empty space
  .  MA20 / -  MA60 / ^v  Bollinger Bands

The chart reads left (past) to right (present).
All analysis and responses must be in English."""


def pass1_prompt(pair: dict) -> str:
    """Pass 1: Show X chart only and request hypothesis generation.
    Force LLM to select pattern name from PATTERN_TAXONOMY → ensures meaningful grouping in step 04.
    """
    return f"""## Stock: {pair['name']} ({pair['ticker']})
## Period: {pair['x_start']} ~ {pair['x_end']} (6 months)

### Text Chart
{pair['x_chart']}

### Market State (Metadata)
{pair['x_meta']}

---
Analyze the chart above and write your prediction for the next 10 trading days.

**You MUST select the pattern name from the list below — exact match required:**
{TAXONOMY_STR}

[Output format — respond with JSON only]
{{
  "pattern": "One from the list above (must match exactly)",
  "hypothesis": "Specific hypothesis — include measurable numbers (e.g. volume 1.5x above avg + Bollinger upper breakout → target +5~8%)",
  "predicted_direction": "UP or DOWN or FLAT",
  "predicted_return_pct": number (e.g. 5.0),
  "confidence": "HIGH or MEDIUM or LOW",
  "key_signals": [
    "Signal 1 — include numbers (e.g. RSI=58, price 3% above MA20)",
    "Signal 2",
    "Signal 3"
  ],
  "risk_factors": ["Risk 1 — include numbers", "Risk 2"]
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

### Your Previous Hypothesis
- Pattern: {hypothesis.get('pattern', '?')}
- Hypothesis: {hypothesis.get('hypothesis', '?')}
- Predicted Direction: {hypothesis.get('predicted_direction', '?')} / Predicted Return: {hypothesis.get('predicted_return_pct', '?')}%
- Confidence: {hypothesis.get('confidence', '?')}

---
### Actual Results (Y: {pair['y_start']} ~ {pair['y_end']})
{y_daily_str}

Final Return: {pair['y_return_pct']:+.3f}%
Max Upside: {pair['y_max_up']:+.3f}% / Max Drawdown: {pair['y_max_down']:+.3f}%
Actual Direction: {pair['y_direction']}

---
You now know the actual outcome. Analyze the following.

**IMPORTANT**: Do NOT write a narrative about "why this happened."
Instead, knowing the result, identify the **specific measurable conditions already visible
in the X chart metadata (RSI, volume, MA position, Bollinger Bands)** that predicted this outcome.
These conditions must be directly applicable to new charts for future testing.

[Output format — respond with JSON only]
{{
  "verdict": "SUCCESS or PARTIAL or FAILURE",
  "direction_correct": true or false,
  "verdict_reason": "One sentence — no narrative, numbers only",

  "measurable_success_condition": {{
    "description": "The decisive condition in the X chart that led to {pair['y_direction']}",
    "volume_signal": "Volume condition (e.g. >= 1.5x average / N/A)",
    "rsi_signal": "RSI condition (e.g. between 50~65 / N/A)",
    "ma_signal": "MA position condition (e.g. close above MA20 + MA20 trending up / N/A)",
    "bollinger_signal": "Bollinger condition (e.g. broke middle band expanding toward upper / N/A)"
  }},

  "measurable_failure_condition": {{
    "description": "Conditions in the X chart that would signal the opposite direction (warning signs)",
    "volume_signal": "Volume warning (e.g. declining volume / N/A)",
    "rsi_signal": "RSI warning (e.g. RSI > 70 overbought / N/A)",
    "ma_signal": "MA warning (e.g. close below MA20 + bearish alignment / N/A)",
    "bollinger_signal": "Bollinger warning (e.g. touched upper band then contracted / N/A)"
  }},

  "testable_rule": "IF [condition A with number] AND [condition B with number] THEN [direction] ELSE [opposite direction]"
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
    """Check Ollama server connection"""
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        log.info(f"Ollama connected. Available models: {models}")
        if not any(MODEL.split(":")[0] in m for m in models):
            log.warning(f"Model '{MODEL}' not found. Run `ollama pull {MODEL}` first.")
            return False
        return True
    except Exception as e:
        log.error(f"Ollama connection failed: {e}")
        log.error("Make sure Ollama is running: `ollama serve`")
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
    if not XY_PATH.exists():
        log.error(f"pairs.jsonl not found. Run 02_generate_xy.py first.")
        return

    if not check_ollama():
        return

    processed = get_processed_ids()
    log.info(f"Already processed: {len(processed)} samples")

    with open(XY_PATH, "r", encoding="utf-8") as f:
        pairs = [json.loads(l) for l in f if l.strip()]

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
            raw_p1     = call_ollama(messages_p1)
            hypothesis = extract_json(raw_p1)
            log.info(f"  P1: {hypothesis.get('predicted_direction','?')} "
                     f"{hypothesis.get('predicted_return_pct','?')}% | "
                     f"{hypothesis.get('confidence','?')}")

            # ── Pass 2: Verify & extract lessons ───────────────────────────
            messages_p2 = messages_p1 + [
                {"role": "assistant", "content": raw_p1},
                {"role": "user",      "content": pass2_prompt(pair, hypothesis)},
            ]
            raw_p2  = call_ollama(messages_p2)
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
