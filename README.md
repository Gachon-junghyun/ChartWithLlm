# CLAUDE_CHART_LLM — KOSPI 200 차트 패턴 LLM 학습 파이프라인

## 개요

```
OHLCV → 텍스트 차트(X) + 미래 10일(Y)
  → Ollama 2-pass 추론 (고정 taxonomy 패턴명 + 측정 가능한 수치 조건 추출)
  → 모순 케이스 탐지 → 수치 기반 IF-THEN 룰 사전(MD) 자동 생성
  → 백테스트: 룰 적용 전/후 예측 정확도 비교
```

---

## 폴더 구조

```
CLAUDE_CHART_LLM/
├── 01_download_ohlcv.py    # KOSPI 200 OHLCV 다운로드 (yfinance)
├── 02_generate_xy.py       # 슬라이딩 윈도우 X/Y 페어 생성
├── 03_llm_pipeline.py      # Ollama 2-pass 추론 (고정 패턴 taxonomy + 수치 조건 추출)
├── 04_rule_extractor.py    # 모순 케이스 → 수치 기반 IF-THEN 룰 추출
├── 05_backtest.py          # 룰 검증: 베이스라인 vs 룰 적용 정확도 비교
│
├── ohlcv_cache/            # 종목별 .pkl (OHLCV 캐시)
├── xy_pairs/
│   └── pairs.jsonl         # X/Y 학습 페어
├── llm_output/
│   ├── annotated.jsonl     # LLM 추론 결과
│   └── backtest_result.json# 백테스트 결과 (05 실행 후 생성)
└── rules/
    ├── knowledge_base.md   # 통합 룰 사전 (수치 기반 IF-THEN)
    └── {패턴명}.md         # 패턴별 세부 룰
```

---

## 윈도우 구조

```
◀──── X: 6개월 (126 거래일) ────▶◀── Y: 10 거래일 ──▶
[                                ][                   ]
                                  ↑ shift = 1개월 (21일)
```

- **X**: text_chart.py로 생성한 텍스트 차트 + 메타데이터 (MA20/60, 볼린저, RSI 등)
- **Y**: 이후 10 거래일 OHLCV + 등락률 + 방향 (UP/DOWN/FLAT, ±2% 기준)

---

## 실행 순서

### 1단계. 패키지 설치

```bash
pip install yfinance openpyxl requests --break-system-packages
```

### 2단계. OHLCV 다운로드 (약 10~20분)

```bash
python3 01_download_ohlcv.py
```

- 200종목 × 3년 데이터를 `ohlcv_cache/` 에 pkl로 저장
- 이미 저장된 종목은 자동 스킵 (재시작 안전)

### 3단계. X/Y 페어 생성

```bash
python3 02_generate_xy.py
```

- 예상 출력: **200종목 × 약 25~30 윈도우 ≈ 5,000~6,000개 페어**
- `xy_pairs/pairs.jsonl` 에 저장

### 4단계. Ollama 실행 확인

```bash
# 모델 다운로드 (최초 1회)
ollama pull gemma3:27b

# 서버 실행 (별도 터미널)
ollama serve
```

### 5단계. LLM 2-pass 추론 (가장 오래 걸림)

```bash
python3 03_llm_pipeline.py
```

- **Pass 1**: X 차트 → 가설 생성 (패턴명, 예측 방향, 신뢰도)
- **Pass 2**: X + Y → 검증, 실패 이유, 교훈 추출
- `llm_output/annotated.jsonl` 에 저장
- 중간 저장 지원 (10건마다) → 중단 후 재시작 가능

### 6단계. 룰 추출 → 지식베이스 MD 생성

```bash
python3 04_rule_extractor.py
```

- 고정 taxonomy 기준으로 패턴 그룹핑 (패턴명 불일치 문제 해결)
- 같은 패턴인데 성공/실패가 엇갈린 케이스 쌍 탐지
- LLM이 "X에서 미리 읽을 수 있었던 수치 조건" 추출 → IF-THEN 룰 생성
- `rules/knowledge_base.md` 로 통합

### 7단계. 백테스트 — 룰 검증

```bash
python3 05_backtest.py
```

- annotated.jsonl을 시간순 정렬 후 앞 80%(train) / 뒤 20%(test) 분리
- test set에서 두 가지 예측 실행:
  - **베이스라인**: 룰 없이 LLM 단독 예측
  - **룰 적용**: knowledge_base.md를 컨텍스트로 제공 후 예측
- 실제 y_direction과 비교 → 방향 정확도 측정
- Majority class 기준선 대비 룰의 실질 효과 확인
- 결과: `llm_output/backtest_result.json`

> **이 단계가 없으면 생성된 룰이 노이즈인지 시그널인지 알 수 없습니다.**

---

## 출력 샘플 (pairs.jsonl)

```json
{
  "id": "005930_2024-01-02",
  "ticker": "005930",
  "name": "삼성전자",
  "x_start": "2023-07-03",
  "x_end": "2024-01-02",
  "y_start": "2024-01-03",
  "y_end": "2024-01-16",
  "x_chart": "...(텍스트 차트)...",
  "x_meta": "이평선: MA20 위(...) | ...",
  "y_return_pct": 3.45,
  "y_max_up": 5.12,
  "y_max_down": -1.23,
  "y_direction": "UP",
  "y_daily": [{"day": 1, "date": "2024-01-03", ...}, ...]
}
```

---

## LLM 추론 흐름 (annotated.jsonl 추가 필드)

```json
{
  "hypothesis": {
    "pattern": "박스권 돌파",
    "predicted_direction": "UP",
    "predicted_return_pct": 5.0,
    "confidence": "MEDIUM",
    "key_signals": ["거래량 증가", "상단 저항 돌파"],
    "risk_factors": ["RSI 과매수"]
  },
  "verdict": {
    "verdict": "FAILURE",
    "verdict_reason": "거래량 없이 저항 돌파 시도, 다음날 되돌림",
    "lesson": "거래량 동반 없는 박스권 돌파는 false breakout 가능성 높음",
    "condition_success": "전일 대비 거래량 1.5배 이상 + 종가 돌파",
    "condition_failure": "거래량 보합 + 장중 돌파 후 종가 복귀"
  }
}
```

---

## 주요 설정값 (각 파일 상단)

| 파일 | 설정 | 기본값 | 설명 |
|------|------|--------|------|
| 01 | PERIOD | 3y | 다운로드 기간 |
| 01 | WORKERS | 8 | 병렬 다운로드 수 |
| 02 | X_DAYS | 126 (6개월) | 입력 차트 길이 |
| 02 | Y_DAYS | 10 | 예측 대상 기간 |
| 02 | SHIFT | 21 (1개월) | 슬라이딩 윈도우 이동 간격 |
| 02 | FLAT_THR | ±2% | FLAT 판정 임계값 |
| 03 | MODEL | gemma3:27b | Ollama 모델 |
| 03 | PATTERN_TAXONOMY | 21개 고정 패턴 | 자유 생성 대신 목록에서 선택 |
| 04 | MAX_CONFLICT_PAIRS | 5쌍/패턴 | 패턴당 최대 모순 쌍 수 |
| 05 | TEST_RATIO | 0.2 | 테스트셋 비율 (시간순 뒤 20%) |
| 05 | MAX_TEST | 100 | 최대 테스트 샘플 수 |

## 설계 원칙

1. **패턴 고정 taxonomy**: Pass 1에서 자유 생성 대신 21개 패턴 중 하나 선택
   → 04 grouping이 의미 있는 묶음을 만들 수 있음

2. **수치 기반 조건 추출**: Pass 2에서 "왜 됐냐" 서사 금지
   → "RSI < 65 + 거래량 1.5배 이상" 같은 테스트 가능한 조건만 추출

3. **시간순 분리 백테스트**: 룰 생성에 쓰인 데이터와 테스트 데이터를 시간으로 분리
   → 룰이 과거 데이터를 외운 것인지, 진짜 예측력이 있는지 확인 가능
