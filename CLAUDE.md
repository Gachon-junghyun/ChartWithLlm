# ChartWithLlm — 프로젝트 지침서

## 프로젝트 한 줄 요약

KOSPI 200 종목의 OHLCV 차트를 **텍스트 그리드**로 변환해 LLM에게 보여주고, 가설 생성 → 사후 검증 → 룰 추출 → 백테스트의 4단계 파이프라인으로 **수치 기반 IF-THEN 매매 룰**을 자동 생성하는 시스템이다.

---

## 아키텍처 전체 흐름

```
01_download_ohlcv.py   → ohlcv_cache/{code}.pkl
        ↓
02_generate_xy.py      → xy_pairs/pairs.jsonl
        ↓
03_llm_pipeline.py     → llm_output/annotated.jsonl   (2-pass Gemini)
        ↓
04_rule_extractor.py   → rules/{패턴명}.md + rules/knowledge_base.md
        ↓
05_backtest.py         → llm_output/backtest_result.json
```

각 단계는 독립 파일로 순서대로 실행되며, 중간 결과를 jsonl/pkl/md로 저장하므로 중단 후 재시작이 안전하다.

---

## 핵심 데이터 구조

### 1. OHLCV 캐시 (`ohlcv_cache/{code}.pkl`)
- pandas DataFrame, 컬럼: `open high low close volume` (소문자)
- yfinance에서 3년치 일봉으로 다운로드 (`auto_adjust=True`)
- 200종목 기준 코드 6자리 zero-pad (예: `005930.pkl`)

### 2. X/Y 페어 (`xy_pairs/pairs.jsonl`)

각 행이 하나의 학습 샘플:

```json
{
  "id":           "005930_2024-01-02",
  "ticker":       "005930",
  "name":         "삼성전자",
  "x_start":      "2023-07-03",
  "x_end":        "2024-01-02",
  "y_start":      "2024-01-03",
  "y_end":        "2024-01-16",
  "x_chart":      "(텍스트 차트 문자열)",
  "x_meta":       "이평선: MA5 위(...) | ...",
  "y_return_pct": 3.45,
  "y_max_up":     5.12,
  "y_max_down":  -1.23,
  "y_direction":  "UP",
  "y_daily":      [{"day":1, "date":"2024-01-03", ...}, ...]
}
```

**윈도우 구조**:
- X: 126 거래일 (6개월) — 입력 차트
- Y: 10 거래일 — 예측 대상
- SHIFT: 21 거래일 (1개월)씩 슬라이딩 → 종목당 약 25~30개 페어

### 3. LLM 추론 결과 (`llm_output/annotated.jsonl`)

pairs.jsonl의 모든 필드 + 아래 추가:

```json
{
  "hypothesis": {
    "pattern":               "Bollinger Squeeze Breakout",
    "hypothesis":            "수축 후 상단 돌파 → +5~8% 예상",
    "predicted_direction":   "UP",
    "predicted_return_pct":  6.0,
    "confidence":            "MEDIUM",
    "key_signals":           ["RSI 58", "볼린저 수축 중", "MA20 위"],
    "risk_factors":          ["RSI 과매수 근접"]
  },
  "verdict": {
    "verdict":                      "FAILURE",
    "direction_correct":            false,
    "verdict_reason":               "거래량 없이 저항 돌파 → 다음날 되돌림",
    "measurable_success_condition": { "volume_signal":"...", "rsi_signal":"...", ... },
    "measurable_failure_condition": { ... },
    "testable_rule":                "IF [조건A] AND [조건B] THEN UP ELSE DOWN"
  }
}
```

### 4. 룰 사전 (`rules/knowledge_base.md` 및 개별 MD)

패턴별 IF-THEN 룰이 마크다운으로 정리됨. 구조:
- 공통 성공 수치 조건 (거래량 / RSI / MA / 볼린저 각각)
- 공통 실패 수치 조건
- 예외 케이스
- 최종 IF-THEN 룰 (테스트 가능 형식)

---

## text_chart.py — 핵심 모듈

### 캔들 문자 체계

| 문자 | 의미 |
|------|------|
| `█`  | 양봉 몸통 (종가 ≥ 시가) |
| `░`  | 음봉 몸통 (종가 < 시가) |
| `─`  | 도지 (시가 ≈ 종가 ±0.01%) |
| `│`  | 꼬리 (고가/저가 영역) |
| `0`  | 빈칸 |
| `.`  | MA20 오버레이 |
| `-`  | MA60 오버레이 |
| `^`/`v` | 볼린저 상단/하단 |

**규칙**: 캔들 문자(`█ ░ ─ │`)는 지표보다 항상 우선 — 지표 오버레이가 캔들을 덮지 않는다.

### 주요 함수

```python
# 1. 텍스트 차트 생성
chart_str = plot_text_chart(
    df,                          # OHLCV DataFrame
    rows=25,                     # 가격 차트 높이
    cols=126,                    # 표시할 최근 캔들 수 (None=전체)
    vol_rows=5,                  # 거래량 차트 높이
    indicators={"MA20": ..., "BB_upper": ...},
    show_meta=False,
)

# 2. 지표 생성 헬퍼
inds = {**add_ma(df, [20, 60]), **add_bollinger(df)}
rsi_lines = add_rsi_line(df, levels=[30, 70])

# 3. 메타데이터 텍스트 생성
meta = generate_metadata(df)
# → "이평선: MA5 위(75000) | MA20 위(73000) | ...\n볼린저: 밴드폭 12.3% (수축 중 ▼)\n..."
```

### generate_metadata() 출력 항목

- **이평선**: MA5/20/60/120 기준 위/아래 + 현재값
- **볼린저**: 밴드폭(%) + 수축/확장 방향
- **거래량**: 최근 5일 vs 이전 5일 비율 (급증/증가/보합/감소/급감)
- **모멘텀**: 5일/20일 등락률
- **RSI(14)**: 수치 + 과매수/과매도/중립 판정
- **캔들힌트**: 도지, 양→음 전환, 음→양 전환

---

## 패턴 Taxonomy (고정 21개)

03_llm_pipeline.py의 Pass 1에서 LLM이 반드시 아래 목록 중 하나를 선택한다. 자유 생성 금지.

**설계 원칙: 패턴명은 가격 구조(모양)만 기술. 미래 방향 정보 포함 금지.**
"Breakout / Bounce / Correction" 같은 방향 암시 단어를 제거해 LLM이 이름이 아닌
실제 차트 그리드를 읽어서 방향을 독립적으로 판단하게 강제한다.

```
Trend Pullback Zone              # 상승추세 중 조정 구간 (재개 여부 미정)
Range Consolidation Upper Test   # 박스권 상단 테스트 (돌파/실패 미정)
Range Consolidation Lower Test   # 박스권 하단 테스트 (이탈/지지 미정)
Ascending Triangle               # 저점 상승 + 고점 수평 수렴
Descending Triangle              # 고점 하락 + 저점 수평 수렴
Head and Shoulders               # 머리어깨형 천장 구조
Inverse Head and Shoulders       # 역머리어깨형 바닥 구조
Double Bottom                    # 이중 바닥 (W형)
Double Top                       # 이중 천장 (M형)
MA Bullish Cross                 # MA20이 MA60 위로 교차한 상태
MA Bearish Cross                 # MA20이 MA60 아래로 교차한 상태
Post Sharp Decline Zone          # 단기 급락 직후 구간 (반등/지속 미정)
Post Sharp Rally Zone            # 단기 급등 직후 구간 (조정/지속 미정)
Rising Channel                   # 고점·저점 모두 우상향 채널
Falling Channel                  # 고점·저점 모두 우하향 채널
Support Level Test               # 수평 지지선 인근 (지지/이탈 미정)
Resistance Level Test            # 수평 저항선 인근 (돌파/실패 미정)
Bollinger Band Squeeze           # 볼린저 밴드 수축 구간 (방향 미정)
RSI Overbought Zone              # RSI 70 초과 상태 (조정/지속 미정)
RSI Oversold Zone                # RSI 30 미만 상태 (반등/추가하락 미정)
Other
```

이 고정 taxonomy가 없으면 04_rule_extractor의 패턴 그룹핑이 의미없는 조각으로 쪼개진다.

### Pass 1 프롬프트 3단계 구조

1. **STEP 1 — Visual reading**: 차트 그리드를 이미지처럼 읽어 관찰 사실만 기술
   (가격 궤적, 최근 20개 캔들 색상, 거래량 방향, MA 위치, 볼린저 상태)
2. **STEP 2 — Structure identification**: 관찰 결과로 패턴명 선택 (방향과 무관)
3. **STEP 3 — Direction prediction**: Step 1 관찰에만 근거해 방향 예측 (패턴명 참조 금지)

`visual_reading` 필드가 JSON에 추가됨 — 차트를 실제로 읽었는지 추적 가능.

---

## LLM 2-Pass 추론 설계

### Pass 1 — 가설 생성 (X만 보여줌)
- 입력: 텍스트 차트 + 메타데이터
- 출력: 패턴명(taxonomy에서 선택) + 예측 방향 + 수치 근거

### Pass 2 — 사후 검증 (X + 실제 Y 결과 모두 보여줌)
- 입력: Pass 1 결과 + 실제 Y 10일치 OHLCV
- 출력: 성공/실패 판정 + **X에서 미리 읽을 수 있었던 측정 조건** 추출
- 핵심: "왜 됐냐" 서사 금지 — `RSI < 65 + 거래량 1.5배` 같은 테스트 가능한 수치 조건만

### 체크포인트 저장
- 10건마다 `annotated.jsonl`에 append 저장
- 이미 처리된 id는 스킵 → 중단 후 안전하게 재시작 가능
- `--limit N` 옵션으로 N개만 처리 가능 (미처리 중 랜덤 추출)

---

## 04_rule_extractor — 룰 생성 로직

1. annotated.jsonl 로드 → `hypothesis.pattern` 기준 그룹핑
2. 같은 패턴에서 `verdict=SUCCESS` vs `verdict=FAILURE` 쌍 탐지 (최대 5쌍/패턴)
3. 각 쌍에서 LLM이 "결정적 수치 차이"를 추출 → IF-THEN 룰 1개
4. 여러 쌍의 룰을 다시 LLM으로 통합 → 패턴별 최종 룰
5. `rules/{패턴명}.md` 개별 저장 + `rules/knowledge_base.md` 통합

**모순 쌍이 없는 패턴**(전부 성공 또는 전부 실패)은 룰 생성 건너뜀.

---

## 05_backtest — 룰 유효성 검증

- **시간순 분리**: annotated.jsonl을 x_end 기준 정렬 → 앞 80% train, 뒤 20% test
- **두 가지 예측**: (A) 룰 없이 LLM 단독, (B) knowledge_base.md 컨텍스트 제공
- **평가 지표**: 방향 정확도 (UP/DOWN/FLAT 3분류), Majority class 기준선 대비 효과
- **결과**: `llm_output/backtest_result.json`

> 이 단계가 없으면 생성된 룰이 노이즈인지 시그널인지 알 수 없다.

---

## 주요 설정값

| 파일 | 변수 | 기본값 | 의미 |
|------|------|--------|------|
| 01 | `PERIOD` | `"3y"` | yfinance 다운로드 기간 |
| 01 | `WORKERS` | `8` | 병렬 다운로드 스레드 수 |
| 02 | `X_DAYS` | `126` | 입력 차트 길이 (거래일) |
| 02 | `Y_DAYS` | `10` | 예측 대상 기간 |
| 02 | `SHIFT` | `21` | 슬라이딩 윈도우 이동 간격 |
| 02 | `FLAT_THR` | `2.0` | ±2% 미만이면 FLAT 판정 |
| 03 | `MODEL` | `"gemini-2.5-flash"` | Gemini 모델명 |
| 03 | `BATCH_SAVE` | `10` | 체크포인트 저장 주기 |
| 04 | `MAX_CONFLICT_PAIRS` | `5` | 패턴당 최대 모순 쌍 수 |
| 05 | `TEST_RATIO` | `0.2` | 테스트셋 비율 |
| 05 | `MAX_TEST` | `100` | 최대 테스트 샘플 수 |

---

## 환경 설정

```bash
# 패키지 설치
pip install yfinance openpyxl google-genai pandas numpy --break-system-packages

# API 키 설정 (03, 04, 05 실행 전 필수)
export GEMINI_API_KEY=your_key_here

# 전체 파이프라인 실행
bash run_all.sh
```

`.env` 파일에 `GEMINI_API_KEY=...` 형태로 저장해도 된다.

---

## 데이터 파일 현황

- `data/kospi200.xlsx`: KOSPI 200 종목 코드(A열) + 종목명(B열), 2행부터 시작
- `ohlcv_cache/`: 약 170+ 종목 .pkl 캐시 존재
- `xy_pairs/pairs.jsonl`: 생성된 X/Y 페어
- `llm_output/annotated.jsonl`: LLM 2-pass 추론 결과 (110개 샘플 처리됨)
- `rules/knowledge_base.md`: 현재 7개 패턴 룰 생성됨 (Bounce After Sharp Decline, Bollinger Squeeze Breakout, RSI Oversold Bounce, RSI Overbought Correction, Resistance Level Breakout, Support Level Bounce, Other)

---

## 설계 원칙 (변경 시 주의)

1. **패턴 고정 taxonomy** — PATTERN_TAXONOMY 리스트를 변경하면 기존 annotated.jsonl과 새 결과가 그룹핑 단계에서 섞여 의미 없는 룰이 생성된다. 변경 시 annotated.jsonl을 처음부터 재생성해야 한다.

2. **수치 조건만 추출** — Pass 2 프롬프트는 서사적 설명을 명시적으로 금지한다. "거래량이 낮아서 실패했다" 같은 표현이 나오면 프롬프트 개선 필요.

3. **시간순 백테스트** — 룰 생성에 쓰인 데이터와 테스트 데이터를 반드시 시간으로 분리해야 한다. `all_samples.sort(key=lambda x: x["x_end"])` 이후 split이 핵심.

4. **캔들 문자 우선순위** — `_CANDLE_CHARS` 집합에 속한 문자는 지표 오버레이로 덮지 않는다. text_chart.py의 이 규칙을 건드리면 차트 시각적 정확도가 깨진다.

---

## 자주 발생하는 문제

| 증상 | 원인 | 해결 |
|------|------|------|
| `pairs.jsonl` 비어있음 | `ohlcv_cache/`에 pkl 없음 | `01_download_ohlcv.py` 먼저 실행 |
| `annotated.jsonl` 비어있음 | pairs.jsonl 없음 또는 API 키 없음 | `02` → `export GEMINI_API_KEY` → `03` 순서 확인 |
| 04에서 "모순 쌍 없음" 패턴만 출력 | annotated 샘플이 너무 적음 | `03`에서 `--limit` 늘리거나 전체 실행 |
| yfinance MultiIndex 오류 | 단일 종목도 MultiIndex 반환하는 경우 | 각 파일 상단의 `raw.columns.get_level_values(0)` 처리로 해결됨 |
| Gemini `thinking_budget=0` 오류 | 구버전 google-genai | `pip install --upgrade google-genai` |
