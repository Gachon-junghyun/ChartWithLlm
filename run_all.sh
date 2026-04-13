#!/bin/bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# run_all.sh  —  KOSPI 200 차트 패턴 LLM 파이프라인 전체 실행
# Mac 터미널에서: bash run_all.sh
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

set -e   # 오류 발생 시 중단

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " KOSPI 200 차트 LLM 파이프라인 시작"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ── 사전 확인 ─────────────────────────────────────────────────

# 1. text_chart.py 위치 확인
TC_PATH="../NEW_INDICATOR/text_chart.py"
if [ ! -f "$TC_PATH" ]; then
    echo "❌ text_chart.py 없음: $TC_PATH"
    echo "   CLAUDE_CHART_LLM 폴더가 NEW_INDICATOR 와 같은 위치에 있는지 확인하세요."
    exit 1
fi
echo "✓ text_chart.py 확인"

# 2. KOSPI 200 xlsx 확인
XLSX="../NEW_INDICATOR/kospi_screener/kospi200.xlsx"
if [ ! -f "$XLSX" ]; then
    echo "❌ kospi200.xlsx 없음: $XLSX"
    exit 1
fi
echo "✓ kospi200.xlsx 확인"

# 3. Ollama 실행 중 확인
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo ""
    echo "⚠️  Ollama 가 실행 중이지 않습니다."
    echo "   별도 터미널에서 먼저 실행하세요:"
    echo "   $ ollama serve"
    echo ""
    echo "   03, 04 단계는 Ollama 없이 스킵됩니다."
    OLLAMA_READY=0
else
    echo "✓ Ollama 실행 중"
    OLLAMA_READY=1
fi

echo ""

# ── Step 1: OHLCV 다운로드 ───────────────────────────────────
echo "━━━ Step 1: OHLCV 다운로드 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 01_download_ohlcv.py
echo ""

# ── Step 2: X/Y 페어 생성 ────────────────────────────────────
echo "━━━ Step 2: X/Y 페어 생성 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 02_generate_xy.py
echo ""

# ── Step 3: LLM 추론 ─────────────────────────────────────────
if [ "$OLLAMA_READY" -eq 1 ]; then
    echo "━━━ Step 3: LLM 2-pass 추론 ━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    python3 03_llm_pipeline.py
    echo ""

    # ── Step 4: 룰 추출 ──────────────────────────────────────
    echo "━━━ Step 4: 룰 추출 → 지식베이스 MD ━━━━━━━━━━━━━━━━━━"
    python3 04_rule_extractor.py
    echo ""
else
    echo "━━━ Step 3, 4: Ollama 미실행으로 스킵 ━━━━━━━━━━━━━━━━━"
    echo "   나중에 수동 실행:"
    echo "   $ python3 03_llm_pipeline.py"
    echo "   $ python3 04_rule_extractor.py"
    echo ""
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " 완료!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
