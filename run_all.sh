#!/bin/bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# run_all.sh  —  KOSPI 200 차트 패턴 LLM 파이프라인 전체 실행
#
# 사용법:
#   bash run_all.sh              # 전체 실행
#   bash run_all.sh --test       # 1배치(10건)만 실행 (동작 확인용)
#   bash run_all.sh --limit 30   # LLM 추론 N건만 실행
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

set -e   # 오류 발생 시 중단

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── 인자 파싱 ──────────────────────────────────────────────────
LIMIT_ARG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --test)
            LIMIT_ARG="--limit 10"
            echo "🔍 테스트 모드: LLM 추론 10건만 실행합니다"
            shift
            ;;
        --limit)
            LIMIT_ARG="--limit $2"
            echo "🔍 제한 모드: LLM 추론 $2건만 실행합니다"
            shift 2
            ;;
        *)
            echo "알 수 없는 옵션: $1"
            echo "사용법: bash run_all.sh [--test] [--limit N]"
            exit 1
            ;;
    esac
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " KOSPI 200 차트 LLM 파이프라인 시작"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ── 사전 확인 ─────────────────────────────────────────────────

# text_chart.py 위치 확인
if [ ! -f "$SCRIPT_DIR/text_chart.py" ]; then
    echo "❌ text_chart.py 없음"
    exit 1
fi
echo "✓ text_chart.py 확인"

# KOSPI 200 xlsx 확인
if [ ! -f "$SCRIPT_DIR/data/kospi200.xlsx" ]; then
    echo "❌ data/kospi200.xlsx 없음"
    exit 1
fi
echo "✓ kospi200.xlsx 확인"

# GEMINI_API_KEY 확인
if [ -z "${GEMINI_API_KEY}" ] && [ -z "${GOOGLE_API_KEY}" ]; then
    echo ""
    echo "❌ GEMINI_API_KEY 환경변수가 설정되지 않았습니다."
    echo "   실행 전: export GEMINI_API_KEY=your_key_here"
    exit 1
fi
echo "✓ GEMINI_API_KEY 확인"

echo ""

# ── Step 1: OHLCV 다운로드 ───────────────────────────────────
echo "━━━ Step 1: OHLCV 다운로드 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 01_download_ohlcv.py
echo ""

# ── Step 2: X/Y 페어 생성 ────────────────────────────────────
echo "━━━ Step 2: X/Y 페어 생성 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 02_generate_xy.py
echo ""

# ── Step 3: LLM 2-pass 추론 ──────────────────────────────────
echo "━━━ Step 3: LLM 2-pass 추론 ━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 03_llm_pipeline.py $LIMIT_ARG
echo ""

# ── Step 4: 룰 추출 → 지식베이스 MD ─────────────────────────
# 테스트 모드(--limit)에서도 실행하되, 샘플이 너무 적으면 룰 생성이 안 될 수 있음
echo "━━━ Step 4: 룰 추출 → 지식베이스 MD ━━━━━━━━━━━━━━━━━━━"
python3 04_rule_extractor.py
echo ""

# ── Step 5: 백테스트 ─────────────────────────────────────────
# 테스트 모드에서는 샘플이 부족해 의미 없으므로 스킵
if [ -z "$LIMIT_ARG" ]; then
    echo "━━━ Step 5: 백테스트 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    python3 05_backtest.py
    echo ""
else
    echo "━━━ Step 5: 백테스트 — 테스트/제한 모드에서 스킵 ━━━━━━━"
    echo "   전체 실행 후: python3 05_backtest.py"
    echo ""
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " 완료!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
