"""
05_backtest.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
룰 검증 파이프라인 — "생성한 룰이 진짜 예측력이 있는가?"

흐름
────
  annotated.jsonl (시간순 정렬)
      │
      ▼
  앞 80%  → train set (룰 생성에 쓰인 데이터, 스킵)
  뒤 20%  → test set  (룰이 본 적 없는 미래 데이터)
      │
      ▼
  각 test pair에 대해 두 가지 예측:
    A) 베이스라인: 룰 없이 LLM 단독 예측
    B) 룰 적용:   knowledge_base.md 를 컨텍스트로 제공 후 예측
      │
      ▼
  실제 y_direction 과 비교 → 방향 정확도 측정
      │
      ▼
  결과 리포트 출력 + backtest_result.json 저장

설정
────
  TEST_RATIO   : 테스트셋 비율 (기본 0.2 = 20%)
  MAX_TEST     : 최대 테스트 샘플 수 (처리 시간 제한)
  MODEL        : Ollama 모델명
"""

import json
import sys
import logging
import random
from pathlib import Path
from datetime import datetime

import requests

# ── 경로 ──────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent
ANNOTATED_PATH = BASE_DIR / "llm_output" / "annotated.jsonl"
KB_PATH        = BASE_DIR / "rules" / "knowledge_base.md"
OUT_PATH       = BASE_DIR / "llm_output" / "backtest_result.json"

# ── 설정 ──────────────────────────────────────────────────────────────────────
TEST_RATIO  = 0.2     # 뒤 20%를 테스트셋으로
MAX_TEST    = 100     # 테스트 샘플 최대 수 (시간 절약)
OLLAMA_URL  = "http://localhost:11434/api/chat"
MODEL       = "gemma3:27b"
TIMEOUT     = 120

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ── Ollama 호출 ────────────────────────────────────────────────────────────────

def call_ollama(messages: list[dict]) -> str:
    payload = {
        "model":    MODEL,
        "messages": messages,
        "stream":   False,
        "options":  {"temperature": 0.1, "top_p": 0.9},
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


def extract_json(text: str) -> dict:
    if "```" in text:
        start = text.find("```")
        end   = text.rfind("```")
        text  = text[start:end].replace("```json", "").replace("```", "").strip()
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start >= 0 and end > start:
        text = text[start:end]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw_response": text}


# ── 예측 프롬프트 ──────────────────────────────────────────────────────────────

SYSTEM_BASE = """당신은 한국 주식 차트 패턴 분석 전문가입니다.
텍스트 차트를 읽고 다음 10 거래일 방향을 예측합니다.

캔들 문자:
  █ 양봉 / ░ 음봉 / ─ 도지 / │ 꼬리 / 0 빈칸
  . MA20 / - MA60 / ^v 볼린저밴드

차트는 왼쪽이 과거, 오른쪽이 현재입니다."""


def predict_prompt(pair: dict) -> str:
    """Y를 절대 포함하지 않는 순수 예측 프롬프트"""
    return f"""## 종목: {pair['name']} ({pair['ticker']})
## 기간: {pair['x_start']} ~ {pair['x_end']}

### 텍스트 차트
{pair['x_chart']}

### 메타데이터
{pair['x_meta']}

---
다음 10 거래일 방향을 예측하세요.

[출력 형식 — JSON만]
{{
  "predicted_direction": "UP 또는 DOWN 또는 FLAT",
  "confidence": "HIGH 또는 MEDIUM 또는 LOW",
  "reason": "근거 1문장"
}}"""


def predict_with_rules_prompt(pair: dict, kb_content: str) -> str:
    """knowledge_base 룰을 컨텍스트로 제공한 예측 프롬프트"""
    return f"""## 참고 룰 사전 (과거 케이스에서 추출된 조건 기반 룰)
{kb_content[:3000]}  ← 토큰 제한으로 앞부분만 사용

---
## 종목: {pair['name']} ({pair['ticker']})
## 기간: {pair['x_start']} ~ {pair['x_end']}

### 텍스트 차트
{pair['x_chart']}

### 메타데이터
{pair['x_meta']}

---
위 룰 사전을 참고하여 이 차트의 패턴을 파악하고 다음 10 거래일 방향을 예측하세요.
어떤 룰을 적용했는지 명시하세요.

[출력 형식 — JSON만]
{{
  "predicted_direction": "UP 또는 DOWN 또는 FLAT",
  "confidence": "HIGH 또는 MEDIUM 또는 LOW",
  "matched_rule": "적용한 룰 이름 또는 패턴 (없으면 null)",
  "reason": "근거 1문장"
}}"""


# ── 평가 유틸 ──────────────────────────────────────────────────────────────────

def direction_correct(predicted: str, actual: str) -> bool:
    """방향 일치 여부 — FLAT은 UP/DOWN 모두 틀린 것으로 처리"""
    return predicted == actual


def summarize(results: list[dict], label: str) -> dict:
    """정확도 집계"""
    total   = len(results)
    correct = sum(1 for r in results if r.get("correct", False))
    high_conf = [r for r in results if r.get("confidence") == "HIGH"]
    high_correct = sum(1 for r in high_conf if r.get("correct", False))

    # 방향별 정확도
    for direction in ["UP", "DOWN", "FLAT"]:
        subset = [r for r in results if r["actual"] == direction]
        correct_sub = sum(1 for r in subset if r.get("correct", False))
        log.info(f"  [{label}] {direction}: {correct_sub}/{len(subset)}"
                 f" ({correct_sub/len(subset)*100:.1f}%)" if subset else
                 f"  [{label}] {direction}: 0/0")

    acc = correct / total * 100 if total else 0
    high_acc = high_correct / len(high_conf) * 100 if high_conf else 0

    return {
        "label":         label,
        "total":         total,
        "correct":       correct,
        "accuracy_pct":  round(acc, 1),
        "high_conf_total":   len(high_conf),
        "high_conf_correct": high_correct,
        "high_conf_accuracy_pct": round(high_acc, 1),
    }


# ── 메인 ──────────────────────────────────────────────────────────────────────

def check_ollama() -> bool:
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        if not any(MODEL.split(":")[0] in m for m in models):
            log.warning(f"'{MODEL}' 없음. `ollama pull {MODEL}` 실행하세요.")
            return False
        return True
    except Exception as e:
        log.error(f"Ollama 연결 실패: {e}")
        return False


def main():
    # ── 데이터 로드 ───────────────────────────────────────────────────────────
    if not ANNOTATED_PATH.exists():
        log.error("annotated.jsonl 없음. 먼저 03_llm_pipeline.py 실행하세요.")
        return

    with open(ANNOTATED_PATH, "r", encoding="utf-8") as f:
        all_samples = [json.loads(l) for l in f if l.strip()]

    if not all_samples:
        log.error("샘플이 없습니다.")
        return

    # 시간순 정렬 (x_end 기준)
    all_samples.sort(key=lambda x: x.get("x_end", ""))
    total = len(all_samples)
    log.info(f"전체 샘플: {total}개")
    log.info(f"기간: {all_samples[0]['x_end']} ~ {all_samples[-1]['x_end']}")

    # train / test 분리
    split_idx  = int(total * (1 - TEST_RATIO))
    test_set   = all_samples[split_idx:]
    log.info(f"Train: {split_idx}개 ({all_samples[0]['x_end']} ~ {all_samples[split_idx-1]['x_end']})")
    log.info(f"Test:  {len(test_set)}개 ({test_set[0]['x_end']} ~ {test_set[-1]['x_end']})")

    # 테스트셋이 너무 크면 랜덤 샘플링
    if len(test_set) > MAX_TEST:
        random.seed(42)
        test_set = random.sample(test_set, MAX_TEST)
        log.info(f"최대 {MAX_TEST}개로 제한 (랜덤 샘플링)")

    # ── knowledge_base 로드 ───────────────────────────────────────────────────
    if not KB_PATH.exists():
        log.warning("knowledge_base.md 없음. 04_rule_extractor.py 먼저 실행하세요.")
        log.warning("룰 없이 베이스라인 테스트만 실행합니다.")
        kb_content = None
    else:
        kb_content = KB_PATH.read_text(encoding="utf-8")
        log.info(f"knowledge_base 로드: {len(kb_content)}자")

    if not check_ollama():
        return

    # ── 예측 실행 ─────────────────────────────────────────────────────────────
    baseline_results = []
    ruled_results    = []

    for i, sample in enumerate(test_set, 1):
        actual = sample.get("y_direction", "UNKNOWN")
        log.info(f"[{i}/{len(test_set)}] {sample['ticker']} {sample['name']} | "
                 f"x_end={sample['x_end']} | 실제={actual}")

        # ── A) 베이스라인 예측 (룰 없음) ───────────────────────────────────
        try:
            raw_base = call_ollama([
                {"role": "system", "content": SYSTEM_BASE},
                {"role": "user",   "content": predict_prompt(sample)},
            ])
            pred_base = extract_json(raw_base)
            base_dir  = pred_base.get("predicted_direction", "UNKNOWN")
            base_ok   = direction_correct(base_dir, actual)
            baseline_results.append({
                "id":         sample["id"],
                "ticker":     sample["ticker"],
                "x_end":      sample["x_end"],
                "actual":     actual,
                "predicted":  base_dir,
                "correct":    base_ok,
                "confidence": pred_base.get("confidence", "?"),
                "reason":     pred_base.get("reason", ""),
            })
            log.info(f"  Base: {base_dir} ({'✓' if base_ok else '✗'}) | {actual}")
        except Exception as e:
            log.error(f"  베이스라인 오류: {e}")

        # ── B) 룰 적용 예측 ───────────────────────────────────────────────
        if kb_content:
            try:
                raw_ruled = call_ollama([
                    {"role": "system", "content": SYSTEM_BASE},
                    {"role": "user",   "content": predict_with_rules_prompt(sample, kb_content)},
                ])
                pred_ruled = extract_json(raw_ruled)
                ruled_dir  = pred_ruled.get("predicted_direction", "UNKNOWN")
                ruled_ok   = direction_correct(ruled_dir, actual)
                ruled_results.append({
                    "id":           sample["id"],
                    "ticker":       sample["ticker"],
                    "x_end":        sample["x_end"],
                    "actual":       actual,
                    "predicted":    ruled_dir,
                    "correct":      ruled_ok,
                    "confidence":   pred_ruled.get("confidence", "?"),
                    "matched_rule": pred_ruled.get("matched_rule"),
                    "reason":       pred_ruled.get("reason", ""),
                })
                log.info(f"  Rule: {ruled_dir} ({'✓' if ruled_ok else '✗'}) | "
                         f"적용룰={pred_ruled.get('matched_rule', 'none')}")
            except Exception as e:
                log.error(f"  룰 예측 오류: {e}")

    # ── 결과 집계 ─────────────────────────────────────────────────────────────
    log.info("\n" + "="*60)
    log.info("[ 백테스트 결과 ]")
    log.info("="*60)

    # 실제 방향 분포 (베이스라인 기준)
    actuals = [r["actual"] for r in baseline_results]
    log.info(f"테스트셋 방향 분포: "
             f"UP={actuals.count('UP')} / DOWN={actuals.count('DOWN')} / FLAT={actuals.count('FLAT')}")

    # 베이스라인 majority-class 정확도 (비교 기준)
    majority = max(set(actuals), key=actuals.count) if actuals else "UP"
    majority_acc = actuals.count(majority) / len(actuals) * 100 if actuals else 0
    log.info(f"Majority class 기준선: {majority} → {majority_acc:.1f}%")
    log.info("")

    base_summary  = summarize(baseline_results, "베이스라인(룰 없음)")
    ruled_summary = summarize(ruled_results,    "룰 적용") if ruled_results else {}

    log.info("")
    log.info(f"베이스라인 정확도: {base_summary['accuracy_pct']}%"
             f" ({base_summary['correct']}/{base_summary['total']})")
    if ruled_summary:
        log.info(f"룰 적용 정확도:   {ruled_summary['accuracy_pct']}%"
                 f" ({ruled_summary['correct']}/{ruled_summary['total']})")
        delta = ruled_summary["accuracy_pct"] - base_summary["accuracy_pct"]
        log.info(f"룰 적용 효과:     {delta:+.1f}%p")

    if ruled_summary:
        log.info("")
        log.info(f"HIGH 신뢰도 베이스라인: {base_summary['high_conf_accuracy_pct']}%"
                 f" ({base_summary['high_conf_correct']}/{base_summary['high_conf_total']})")
        log.info(f"HIGH 신뢰도 룰 적용:   {ruled_summary['high_conf_accuracy_pct']}%"
                 f" ({ruled_summary['high_conf_correct']}/{ruled_summary['high_conf_total']})")

    # 룰 매칭 통계
    if ruled_results:
        matched = [r for r in ruled_results if r.get("matched_rule")]
        log.info(f"\n룰 매칭률: {len(matched)}/{len(ruled_results)}"
                 f" ({len(matched)/len(ruled_results)*100:.1f}%)")
        if matched:
            matched_correct = sum(1 for r in matched if r["correct"])
            log.info(f"룰 매칭 시 정확도: {matched_correct}/{len(matched)}"
                     f" ({matched_correct/len(matched)*100:.1f}%)")

    # ── JSON 저장 ──────────────────────────────────────────────────────────────
    output = {
        "run_at":          datetime.now().strftime("%Y-%m-%d %H:%M"),
        "model":           MODEL,
        "test_set_size":   len(test_set),
        "train_set_size":  split_idx,
        "majority_class":  majority,
        "majority_acc_pct": round(majority_acc, 1),
        "baseline":        base_summary,
        "ruled":           ruled_summary,
        "improvement_pct": round(
            ruled_summary.get("accuracy_pct", 0) - base_summary["accuracy_pct"], 1
        ) if ruled_summary else None,
        "baseline_details": baseline_results,
        "ruled_details":    ruled_results,
    }
    OUT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"\n결과 저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
