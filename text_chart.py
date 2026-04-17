"""
text_chart.py  v2  —  LLM 차트 패턴 학습 특화
══════════════════════════════════════════════════════════════════════════════
OHLCV DataFrame → 캔들 텍스트 차트 + 자동 메타데이터 + 학습 샘플 생성기

설계 목적
─────────
눌림목·돌파·삼각수렴 등 차트 패턴을 LLM에게 텍스트로 학습시키기 위한
시각화 + 데이터 준비 도구.

LLM이 읽을 수 있는 2D 그리드로 OHLC 전체 정보를 보존하고,
패턴 구간 라벨 + 자동 메타데이터를 함께 제공해 학습 쌍을 구성한다.

캔들 문자 체계
──────────────
  █  양봉 몸통 (종가 ≥ 시가)
  ░  음봉 몸통 (종가 < 시가)
  ─  도지    (시가 ≈ 종가 ±0.01%)
  │  꼬리    (고가/저가 영역)
  0  빈칸

주요 함수
─────────
  plot_text_chart()       메인 차트 생성
  add_ma()                단순 이동평균 딕셔너리
  add_bollinger()         볼린저 밴드 딕셔너리
  add_rsi_line()          RSI 수평선 오버레이
  generate_metadata()     자동 메타데이터 텍스트
  make_training_samples() LLM 학습 샘플 (JSONL 저장 가능)

빠른 시작
─────────
    import yfinance as yf
    from text_chart import plot_text_chart, add_ma, add_bollinger

    raw = yf.download("005930.KS", period="6mo", auto_adjust=True)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.columns = raw.columns.str.lower()

    inds = {**add_ma(raw, [20, 60]), **add_bollinger(raw)}
    print(plot_text_chart(raw, rows=25, cols=80,
                          indicators=inds, show_meta=True))
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ══════════════════════════════════════════════════════════════════════════════
#  문자 상수
# ══════════════════════════════════════════════════════════════════════════════

EMPTY     = " "   # 빈칸
WICK      = "│"   # 꼬리 (고가/저가)
BULL_BODY = "█"   # 양봉 몸통
BEAR_BODY = "░"   # 음봉 몸통
DOJI_BODY = "─"   # 도지

VOL_BULL  = "█"   # 양봉 거래량
VOL_BEAR  = "▒"   # 음봉 거래량

DEFAULT_IND_CHARS = [".", "-", "+", "~", "*", "="]

# 캔들 문자는 지표보다 항상 우선 (오버레이 금지)
_CANDLE_CHARS = {WICK, BULL_BODY, BEAR_BODY, DOJI_BODY}

# RSI/OBV 서브차트 문자
RSI_DOT   = "*"   # RSI 점 (중립, 30~70)
RSI_OB    = "+"   # RSI > 70 (과매수) — 가격차트 ^ (BB상단) 과 구분
RSI_OS    = "="   # RSI < 30 (과매도) — 가격차트 v (BB하단) 과 구분
RSI_REF   = "─"   # 30/70 기준선
RSI_MID   = ":"   # 50 중간선 — 가격차트 · (BB중단) 과 구분
OBV_BULL  = "█"   # OBV 양봉 (매수압력)
OBV_BEAR  = "░"   # OBV 음봉 (매도압력)
OBV_ZERO  = "─"   # OBV 중립선

# 통합 차트용 고정 지표 문자 (순서 고정으로 LLM 혼선 방지)
_COMBINED_IND_CHARS = {
    "MA20":     ".",
    "MA60":     "-",
    "BB_upper": "^",
    "BB_mid":   "·",
    "BB_lower": "v",
}


# ══════════════════════════════════════════════════════════════════════════════
#  내부 유틸리티
# ══════════════════════════════════════════════════════════════════════════════

def _price_to_row(price: float, min_p: float, max_p: float, rows: int) -> int:
    """가격 → 그리드 행 인덱스.  row 0 = 상단 = 고가."""
    p_range = max_p - min_p
    if p_range == 0:
        return rows // 2
    ratio = (price - min_p) / p_range          # 0.0(저) … 1.0(고)
    row   = int((1.0 - ratio) * (rows - 1))    # 뒤집기: 고가 → 위
    return max(0, min(rows - 1, row))


def _row_to_price(row: int, min_p: float, max_p: float, rows: int) -> float:
    """그리드 행 인덱스 → 가격 (_price_to_row 역함수)."""
    ratio = 1.0 - row / max(rows - 1, 1)
    return min_p + ratio * (max_p - min_p)


def _fmt_price(price: float) -> str:
    if abs(price) >= 10_000: return f"{price:,.0f}"
    if abs(price) >= 1:      return f"{price:.2f}"
    return f"{price:.5f}"


def _fmt_vol(vol: float) -> str:
    if vol >= 1e9: return f"{vol/1e9:.1f}B"
    if vol >= 1e6: return f"{vol/1e6:.1f}M"
    if vol >= 1e3: return f"{vol/1e3:.1f}K"
    return f"{vol:.0f}"


def _norm(df: pd.DataFrame) -> pd.DataFrame:
    """컬럼명 소문자 정규화."""
    out = df.copy()
    out.columns = [c.lower() for c in out.columns]
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  캔들 렌더링
# ══════════════════════════════════════════════════════════════════════════════

def _render_candle(
    col: list[str],
    o: float, h: float, l: float, c: float,
    min_p: float, max_p: float, rows: int,
) -> None:
    """
    단일 캔들을 col(길이=rows 리스트)에 인플레이스로 그린다.

    그리드 좌표계 (row 0 = 상단 = 고가)
    ─────────────────────────────────────
    high_row  ←  고가
    ~~~ 상단 꼬리 │ ~~~
    body_top  ←  max(open, close)
    ~~~ 몸통 █/░/─ ~~~
    body_bot  ←  min(open, close)
    ~~~ 하단 꼬리 │ ~~~
    low_row   ←  저가
    """
    high_row = _price_to_row(h, min_p, max_p, rows)
    low_row  = _price_to_row(l, min_p, max_p, rows)
    body_top = _price_to_row(max(o, c), min_p, max_p, rows)
    body_bot = _price_to_row(min(o, c), min_p, max_p, rows)

    is_doji  = abs(c - o) / (abs(o) + 1e-10) < 0.0001
    body_ch  = DOJI_BODY if is_doji else (BULL_BODY if c >= o else BEAR_BODY)

    for r in range(rows):
        if body_top <= r <= body_bot:          # 몸통
            col[r] = body_ch
        elif high_row <= r < body_top:         # 상단 꼬리
            col[r] = WICK
        elif body_bot < r <= low_row:          # 하단 꼬리
            col[r] = WICK
        # else: EMPTY 그대로


# ══════════════════════════════════════════════════════════════════════════════
#  지표 팩토리
# ══════════════════════════════════════════════════════════════════════════════

def add_ma(df: pd.DataFrame, windows: list[int]) -> Dict[str, pd.Series]:
    """
    단순 이동평균(SMA) 딕셔너리 생성.

    Example
    -------
    indicators = add_ma(raw, [20, 60, 120])
    """
    _df = _norm(df)
    return {f"MA{w}": _df["close"].rolling(w).mean() for w in windows}


def add_bollinger(
    df: pd.DataFrame, window: int = 20, n_std: float = 2.0
) -> Dict[str, pd.Series]:
    """
    볼린저 밴드 딕셔너리 생성.

    Returns: {"BB_upper", "BB_mid", "BB_lower"}
    """
    _df = _norm(df)
    mid = _df["close"].rolling(window).mean()
    std = _df["close"].rolling(window).std()
    return {
        "BB_upper": mid + n_std * std,
        "BB_mid":   mid,
        "BB_lower": mid - n_std * std,
    }


def add_rsi_line(
    df: pd.DataFrame, window: int = 14, levels: list[float] = [30.0, 70.0]
) -> Dict[str, pd.Series]:
    """
    RSI 수평선을 가격 스케일로 변환해 오버레이용 딕셔너리로 반환.
    RSI 50 → 중간 가격,  RSI 0/100 → 저가/고가 에 선형 매핑.

    LLM이 "RSI 30 아래 과매도 구간" 같은 시각적 맥락을 학습하도록 돕는 용도.
    실제 RSI 수치는 generate_metadata()가 텍스트로 별도 제공.
    """
    _df   = _norm(df)
    delta = _df["close"].diff()
    gain  = delta.clip(lower=0).rolling(window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = 100 - 100 / (1 + rs)

    close     = _df["close"]
    price_min = close.rolling(window * 3, min_periods=1).min()
    price_max = close.rolling(window * 3, min_periods=1).max()

    result = {}
    for lvl in levels:
        # RSI 값 → 가격 스케일 선형 매핑 (0→price_min, 100→price_max)
        price_line = price_min + (lvl / 100.0) * (price_max - price_min)
        result[f"RSI{lvl:.0f}"] = price_line

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  자동 메타데이터 생성
# ══════════════════════════════════════════════════════════════════════════════

def generate_metadata(df: pd.DataFrame) -> str:
    """
    차트 구간의 시장 상태를 텍스트로 자동 생성.
    LLM 프롬프트에 차트 텍스트와 함께 붙여 문맥을 제공한다.

    항목: MA 상태 / 볼린저 수축·확장 / 거래량 추세 / 모멘텀 / RSI
    """
    _df   = _norm(df)
    close = _df["close"]
    vol   = _df["volume"] if "volume" in _df.columns else None
    lines: list[str] = []
    last  = float(close.iloc[-1])

    # ── MA 상태 ──────────────────────────────────────────────────────────────
    ma_parts: list[str] = []
    for w in [5, 20, 60, 120]:
        if len(close) >= w:
            ma_val = float(close.rolling(w).mean().iloc[-1])
            state  = "위" if last > ma_val else "아래"
            ma_parts.append(f"MA{w} {state}({_fmt_price(ma_val)})")
    if ma_parts:
        lines.append("이평선: " + " | ".join(ma_parts))

    # ── 볼린저 밴드 수축/확장 ───────────────────────────────────────────────
    if len(close) >= 20:
        mid = close.rolling(20).mean()
        std = close.rolling(20).std()
        bw  = std * 4 / mid * 100      # 밴드폭 (%)
        bw_now  = float(bw.iloc[-1])
        bw_prev = float(bw.iloc[-10]) if len(bw) > 10 else float(bw.iloc[0])
        bb_dir  = "수축 중 ▼" if bw_now < bw_prev else "확장 중 ▲"
        bb_pos  = ""
        upper = float((mid + std * 2).iloc[-1])
        lower = float((mid - std * 2).iloc[-1])
        if last >= upper * 0.98:
            bb_pos = " / 상단 밴드 접근"
        elif last <= lower * 1.02:
            bb_pos = " / 하단 밴드 접근"
        lines.append(f"볼린저: 밴드폭 {bw_now:.1f}% ({bb_dir}){bb_pos}")

    # ── 거래량 추세 ──────────────────────────────────────────────────────────
    if vol is not None and len(vol) >= 10:
        v_rec  = float(vol.iloc[-5:].mean())
        v_prev = float(vol.iloc[-10:-5].mean())
        ratio  = v_rec / (v_prev + 1e-10)
        if   ratio >= 1.5:  vstatus = f"급증 ({ratio:.1f}x) ▲▲"
        elif ratio >= 1.1:  vstatus = f"증가 ({ratio:.1f}x) ▲"
        elif ratio <= 0.5:  vstatus = f"급감 ({ratio:.1f}x) ▼▼"
        elif ratio <= 0.9:  vstatus = f"감소 ({ratio:.1f}x) ▼"
        else:               vstatus = "보합"
        lines.append(f"거래량: {vstatus}")

    # ── 가격 모멘텀 ──────────────────────────────────────────────────────────
    mom_parts: list[str] = []
    for n, label in [(5, "5일"), (20, "20일")]:
        if len(close) >= n:
            ret = (close.iloc[-1] / close.iloc[-n] - 1) * 100
            mom_parts.append(f"{label} {ret:+.1f}%")
    if mom_parts:
        lines.append("모멘텀: " + " | ".join(mom_parts))

    # ── RSI ──────────────────────────────────────────────────────────────────
    if len(close) >= 15:
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss.replace(0, np.nan)
        rsi   = float((100 - 100 / (1 + rs)).iloc[-1])
        if not np.isnan(rsi):
            rsi_label = "과매수 ⚠" if rsi >= 70 else ("과매도 ⚠" if rsi <= 30 else "중립")
            lines.append(f"RSI(14): {rsi:.1f} ({rsi_label})")

    # ── 최근 캔들 패턴 힌트 ──────────────────────────────────────────────────
    if len(_df) >= 3 and all(c in _df.columns for c in ["open", "high", "low", "close"]):
        hints: list[str] = []
        c0 = _df.iloc[-1]   # 최근
        c1 = _df.iloc[-2]
        body0 = abs(float(c0["close"]) - float(c0["open"]))
        wick0 = float(c0["high"]) - float(c0["low"])
        if wick0 > 0 and body0 / wick0 < 0.2:
            hints.append("도지(우유부단)")
        if float(c0["close"]) < float(c0["open"]) and float(c1["close"]) > float(c1["open"]):
            hints.append("양→음 전환")
        if float(c0["close"]) > float(c0["open"]) and float(c1["close"]) < float(c1["open"]):
            hints.append("음→양 전환")
        if hints:
            lines.append("캔들힌트: " + " / ".join(hints))

    return "\n".join(lines) if lines else "(데이터 부족 — 메타데이터 생성 불가)"


# ══════════════════════════════════════════════════════════════════════════════
#  메인 차트 함수
# ══════════════════════════════════════════════════════════════════════════════

def plot_text_chart(
    df: pd.DataFrame,
    rows: int = 30,
    cols: Optional[int] = None,
    vol_rows: int = 6,
    indicators: Optional[Dict[str, pd.Series]] = None,
    indicator_chars: Optional[Dict[str, str]] = None,
    labels: Optional[Dict[str, Tuple[int, int]]] = None,
    show_meta: bool = False,
) -> str:
    """
    OHLCV DataFrame → 캔들 텍스트 차트.

    Parameters
    ----------
    df              : OHLCV DataFrame.  컬럼명 대소문자 무관.
                      필수: open, high, low, close / 선택: volume
    rows            : 가격 차트 높이(행 수).  기본 30.
    cols            : 표시할 최근 캔들 수.  None = 전체.
    vol_rows        : 거래량 차트 높이.  0 = 생략.  기본 6.
    indicators      : 오버레이 지표 dict.
                      {'MA20': Series, 'BB_upper': Series, ...}
                      add_ma() / add_bollinger() 헬퍼로 생성 가능.
    indicator_chars : 각 지표에 쓸 문자 override.
                      {'MA20': '.', 'BB_upper': '^', ...}
                      미지정 시 DEFAULT_IND_CHARS 순환.
    labels          : 패턴 라벨 dict.
                      {'눌림목': (10, 25), '돌파': (26, 30)}
                      인덱스는 슬라이스 후 기준.
    show_meta       : True 이면 하단에 자동 메타데이터 출력.

    Returns
    -------
    str   바로 print() 가능한 멀티라인 문자열.

    전체 레이아웃
    ─────────────
        가격 레이블 | 캔들 그리드 (왼=과거, 오=현재)
        ───────────+──────────────────────────────────
        거래량 레이블 | 볼륨 바 (양봉=█, 음봉=▒)
                    | 패턴 라벨 행 (A, B, C …)
        [캔들·지표 범례]
        [메타데이터]   (show_meta=True 시)
    """
    # ── 0. 전처리 ─────────────────────────────────────────────────────────────
    _df = _norm(df)

    # ── 1. 슬라이싱 ───────────────────────────────────────────────────────────
    if cols is not None:
        _df = _df.iloc[-cols:].reset_index(drop=True)
        if indicators:
            indicators = {
                k: v.iloc[-cols:].reset_index(drop=True)
                for k, v in indicators.items()
            }

    n = len(_df)
    if n == 0:
        raise ValueError(
            "DataFrame이 비어 있습니다. "
            "다운로드 성공 여부와 cols 값을 확인하세요."
        )

    o_arr = _df["open"].values.astype(float)
    h_arr = _df["high"].values.astype(float)
    l_arr = _df["low"].values.astype(float)
    c_arr = _df["close"].values.astype(float)

    # 정규화 기준: 전체 고저가 범위 (종가 기준 X)
    min_p = float(l_arr.min())
    max_p = float(h_arr.max())

    # ── 2. 캔들 그리드 ────────────────────────────────────────────────────────
    grid: list[list[str]] = [[EMPTY] * n for _ in range(rows)]

    for ci in range(n):
        col = [EMPTY] * rows
        _render_candle(col, o_arr[ci], h_arr[ci], l_arr[ci], c_arr[ci],
                       min_p, max_p, rows)
        for r in range(rows):
            grid[r][ci] = col[r]

    # ── 3. 지표 오버레이 ──────────────────────────────────────────────────────
    if indicators:
        if indicator_chars is None:
            indicator_chars = {}
        for idx, (name, series) in enumerate(indicators.items()):
            ch  = indicator_chars.get(name, DEFAULT_IND_CHARS[idx % len(DEFAULT_IND_CHARS)])
            arr = (series.values if hasattr(series, "values")
                   else np.asarray(series)).astype(float)
            for ci in range(min(n, len(arr))):
                v = arr[ci]
                if np.isnan(v):
                    continue
                ri = _price_to_row(v, min_p, max_p, rows)
                if grid[ri][ci] not in _CANDLE_CHARS:  # 캔들은 건드리지 않음
                    grid[ri][ci] = ch

    # ── 4. 가격 축 레이블 폭 ──────────────────────────────────────────────────
    label_row_set = {0, rows // 4, rows // 2, 3 * rows // 4, rows - 1}
    label_prices  = {r: _row_to_price(r, min_p, max_p, rows) for r in label_row_set}
    lw = max(len(_fmt_price(p)) for p in label_prices.values())

    # ── 5. 가격 차트 렌더링 ───────────────────────────────────────────────────
    lines: list[str] = []
    for ri in range(rows):
        row_str = "".join(grid[ri])
        lbl = (_fmt_price(label_prices[ri]).rjust(lw)
               if ri in label_row_set else " " * lw)
        lines.append(f"{lbl} |{row_str}")

    # ── 6. 구분선 ─────────────────────────────────────────────────────────────
    lines.append(" " * lw + "-+" + "-" * n)

    # ── 7. 거래량 차트 ────────────────────────────────────────────────────────
    if vol_rows > 0 and "volume" in _df.columns:
        v_arr = _df["volume"].values.astype(float)
        max_v = v_arr.max()

        vgrid: list[list[str]] = [[" "] * n for _ in range(vol_rows)]
        for ci in range(n):
            if v_arr[ci] <= 0 or max_v == 0:
                continue
            bar    = max(1, round(v_arr[ci] / max_v * vol_rows))
            v_ch   = VOL_BULL if c_arr[ci] >= o_arr[ci] else VOL_BEAR
            for r in range(vol_rows - bar, vol_rows):
                vgrid[r][ci] = v_ch

        vol_max_lbl = _fmt_vol(max_v)
        for ri, vrow in enumerate(vgrid):
            lbl = vol_max_lbl.rjust(lw) if ri == 0 else " " * lw
            lines.append(f"{lbl} |{''.join(vrow)}")

    # ── 8. 패턴 라벨 행 ──────────────────────────────────────────────────────
    if labels:
        _syms       = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        label_row   = [" "] * n
        label_legend: dict[str, str] = {}

        for sym_idx, (pat_name, (s, e)) in enumerate(labels.items()):
            sym = _syms[sym_idx % len(_syms)]
            label_legend[sym] = pat_name
            for ci in range(max(0, s), min(n, e + 1)):
                label_row[ci] = sym

        lines.append(" " * lw + " |" + "".join(label_row))
        leg_str = "  ".join(f"[{sym}]={name}" for sym, name in label_legend.items())
        lines.append(" " * (lw + 2) + leg_str)

    # ── 9. 범례 ───────────────────────────────────────────────────────────────
    c_leg = (f"{BULL_BODY}=양봉  {BEAR_BODY}=음봉  {DOJI_BODY}=도지  "
             f"{WICK}=꼬리  {EMPTY}=빈칸")
    lines.append("")
    lines.append("  캔들: " + c_leg)

    if indicators:
        if indicator_chars is None:
            indicator_chars = {}
        i_parts = [
            f"[{indicator_chars.get(nm, DEFAULT_IND_CHARS[i % len(DEFAULT_IND_CHARS)])}]{nm}"
            for i, nm in enumerate(indicators)
        ]
        lines.append("  지표: " + "  ".join(i_parts))

    # ── 10. 메타데이터 ────────────────────────────────────────────────────────
    if show_meta:
        lines.append("")
        lines.append("── 메타데이터 " + "─" * 44)
        lines.append(generate_metadata(_df))

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  LLM 학습 샘플 생성기
# ══════════════════════════════════════════════════════════════════════════════

def make_training_samples(
    df: pd.DataFrame,
    pattern_labels: Dict[str, List[Tuple[int, int]]],
    context_before: int = 30,
    context_after:  int = 10,
    rows: int = 25,
    vol_rows: int = 5,
    include_indicators: bool = True,
) -> List[Dict]:
    """
    패턴 라벨 구간을 기준으로 LLM 학습 샘플(chart + meta + label)을 생성한다.

    Parameters
    ----------
    df              : 전체 OHLCV DataFrame
    pattern_labels  : 패턴명 → [(절대 인덱스 start, end), ...] 리스트
                      예: {"눌림목": [(50, 70), (120, 135)],
                           "돌파":   [(71, 80)]}
    context_before  : 패턴 시작 전 추가 캔들 수.  기본 30.
    context_after   : 패턴 끝 후 추가 캔들 수.  기본 10.
    rows            : 차트 높이.  기본 25.
    vol_rows        : 거래량 차트 높이.  기본 5.
    include_indicators : True 이면 MA20/MA60/볼린저 자동 추가.

    Returns
    -------
    list[dict]
        각 항목은 아래 키를 가진다:
          "pattern"   : 패턴 이름
          "start_idx" : 절대 start 인덱스
          "end_idx"   : 절대 end 인덱스
          "chart"     : 텍스트 차트 (라벨 표시 포함)
          "meta"      : 메타데이터 텍스트

    JSONL 저장 예시
    ───────────────
        samples = make_training_samples(raw, {"눌림목": [(50, 70)]})
        with open("train.jsonl", "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\\n")

    LLM 프롬프트 조합 예시
    ──────────────────────
        system = "다음 텍스트 차트에서 패턴을 분석하라."
        user   = sample["chart"] + "\\n\\n[시장 상태]\\n" + sample["meta"]
        answer = f"이 차트는 '{sample['pattern']}' 패턴을 보여줍니다."
    """
    _df = _norm(df)
    samples: list[dict] = []

    for pat_name, intervals in pattern_labels.items():
        for abs_start, abs_end in intervals:
            # 슬라이스 범위
            sl_start = max(0, abs_start - context_before)
            sl_end   = min(len(_df), abs_end + context_after + 1)
            win      = _df.iloc[sl_start:sl_end].reset_index(drop=True)

            # 라벨 → 슬라이스 내 상대 좌표
            rel_s = abs_start - sl_start
            rel_e = abs_end   - sl_start
            lbl_map = {pat_name: (rel_s, rel_e)}

            # 지표 자동 생성
            inds: Optional[Dict[str, pd.Series]] = None
            ind_chars: Optional[Dict[str, str]]  = None
            if include_indicators:
                inds, ind_chars = {}, {}
                for w, ch in [(20, "."), (60, "-")]:
                    if len(win) >= w:
                        inds[f"MA{w}"]      = win["close"].rolling(w).mean()
                        ind_chars[f"MA{w}"] = ch
                if len(win) >= 20:
                    bb = add_bollinger(win)
                    inds.update(bb)
                    ind_chars.update({
                        "BB_upper": "^", "BB_mid": "·", "BB_lower": "v"
                    })

            chart_txt = plot_text_chart(
                win,
                rows=rows, cols=None, vol_rows=vol_rows,
                indicators=inds, indicator_chars=ind_chars,
                labels=lbl_map, show_meta=False,
            )
            meta_txt = generate_metadata(win)

            samples.append({
                "pattern":   pat_name,
                "start_idx": abs_start,
                "end_idx":   abs_end,
                "chart":     chart_txt,
                "meta":      meta_txt,
            })

    return samples


# ══════════════════════════════════════════════════════════════════════════════
#  RSI / OBV 서브차트 내부 렌더러
# ══════════════════════════════════════════════════════════════════════════════

def _compute_rsi_arr(close: pd.Series, window: int = 14) -> np.ndarray:
    """pandas 기반 RSI 계산 → numpy array 반환"""
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window).mean()
    rs    = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).values


def _render_rsi_subplot(
    df: pd.DataFrame,
    lw: int,
    n: int,
    rows: int = 7,
    window: int = 14,
) -> list[str]:
    """
    RSI 서브차트를 문자열 라인 목록으로 반환.
    lw: 가격 차트와 동일한 레이블 폭 (정렬용).
    n : 표시할 캔들 수 (이미 슬라이싱된 df 기준).

    문자 체계:
      *  RSI 점 (중립, 30~70)
      ^  RSI > 70 (과매수)
      v  RSI < 30 (과매도)
      ─  30 / 70 기준선
      ·  50 중간선
    """
    _df   = _norm(df)
    close = _df["close"].astype(float)
    rsi   = _compute_rsi_arr(close, window)

    grid = [[EMPTY] * n for _ in range(rows)]

    ob_row  = _price_to_row(70.0, 0.0, 100.0, rows)
    os_row  = _price_to_row(30.0, 0.0, 100.0, rows)
    mid_row = _price_to_row(50.0, 0.0, 100.0, rows)

    # RSI 점 렌더링
    for ci in range(n):
        val = rsi[ci]
        if np.isnan(val):
            continue
        ri = _price_to_row(float(val), 0.0, 100.0, rows)
        ch = RSI_OB if val >= 70 else (RSI_OS if val <= 30 else RSI_DOT)
        grid[ri][ci] = ch

    # 기준선 채우기 (RSI 점 없는 셀만)
    for ref_row, ref_ch in [(ob_row, RSI_REF), (mid_row, RSI_MID), (os_row, RSI_REF)]:
        for ci in range(n):
            if grid[ref_row][ci] == EMPTY:
                grid[ref_row][ci] = ref_ch

    # 레이블 맵
    label_map: dict[int, str] = {}
    for row_idx, lbl in [(0, "100"), (ob_row, " 70"), (mid_row, " 50"),
                          (os_row, " 30"), (rows - 1, "  0")]:
        label_map[row_idx] = lbl

    sep = " " * lw + "-+" + "-" * n
    header = "RSI".rjust(lw) + f"({window})" + " " * max(0, n - len(f"({window})")) + ""

    lines: list[str] = [sep]
    # 헤더 행: 레이블 영역에 "RSI" 표시
    lines.append("RSI".rjust(lw) + " |" + " " * n)
    for ri in range(rows):
        row_str = "".join(grid[ri])
        lbl     = label_map.get(ri, "").rjust(lw)
        lines.append(f"{lbl} |{row_str}")

    return lines


def _render_obv_subplot(
    df: pd.DataFrame,
    lw: int,
    n: int,
    rows: int = 6,
    window: int = 20,
) -> list[str]:
    """
    Rolling OBV 서브차트를 문자열 라인 목록으로 반환.

    Rolling OBV 정의:
      signed_vol[i] = sign(close[i] - close[i-1]) × volume[i]
      obv_roll[i]   = Σ signed_vol[i-window+1 … i]
    → 일정 기간 내 매수/매도 압력 누적치. 전체 OBV가 아닌 국소 모멘텀 표시.

    문자 체계:
      █  양봉 바 (OBV > 0 구간)
      ░  음봉 바 (OBV < 0 구간)
      ─  중립선 (OBV ≈ 0)
    """
    _df = _norm(df)
    if "volume" not in _df.columns:
        return []

    close = _df["close"].astype(float)
    vol   = _df["volume"].astype(float)

    signed_vol = np.sign(close.diff().fillna(0)) * vol
    obv_roll   = pd.Series(signed_vol).rolling(window, min_periods=1).sum().values

    obv_min = float(np.nanmin(obv_roll))
    obv_max = float(np.nanmax(obv_roll))

    if obv_max == obv_min:
        return []

    # 0을 항상 범위에 포함
    scale_min = min(obv_min, 0.0)
    scale_max = max(obv_max, 0.0)

    zero_row = _price_to_row(0.0, scale_min, scale_max, rows)
    grid = [[EMPTY] * n for _ in range(rows)]

    for ci in range(n):
        val = obv_roll[ci]
        if np.isnan(val):
            continue
        val_row = _price_to_row(float(val), scale_min, scale_max, rows)
        ch      = OBV_BULL if val >= 0 else OBV_BEAR
        r_top   = min(zero_row, val_row)
        r_bot   = max(zero_row, val_row)
        for r in range(r_top, r_bot + 1):
            grid[r][ci] = ch

    # 중립선
    for ci in range(n):
        if grid[zero_row][ci] == EMPTY:
            grid[zero_row][ci] = OBV_ZERO

    sep  = " " * lw + "-+" + "-" * n
    lines: list[str] = [sep]
    lines.append(f"OBV{window}d".rjust(lw) + " |" + " " * n)
    for ri in range(rows):
        row_str = "".join(grid[ri])
        lines.append(" " * lw + " |" + row_str)

    return lines


# ══════════════════════════════════════════════════════════════════════════════
#  통합 ASCII 차트 (가격 + 거래량 + RSI + OBV)
# ══════════════════════════════════════════════════════════════════════════════

def plot_combined_chart(
    df: pd.DataFrame,
    rows: int = 20,
    cols: Optional[int] = None,
    vol_rows: int = 4,
    rsi_rows: int = 15,
    rsi_window: int = 14,
    obv_rows: int = 10,
    obv_window: int = 20,
    save_path: Optional[str] = None,
    save_meta: Optional[Dict] = None,
) -> str:
    """
    가격 캔들 + 거래량 + RSI + Rolling OBV 를 하나의 ASCII 차트로 반환.

    Parameters
    ----------
    df          : OHLCV DataFrame (컬럼 대소문자 무관)
    rows        : 가격 차트 높이
    cols        : 표시할 최근 캔들 수 (None = 전체)
    vol_rows    : 거래량 차트 높이
    rsi_rows    : RSI 서브차트 높이
    rsi_window  : RSI 계산 기간 (기본 14)
    obv_rows    : OBV 서브차트 높이
    obv_window  : Rolling OBV 누적 기간 (기본 20)

    차트 구조
    ---------
        가격 레이블 | 캔들 그리드 (MA20/MA60/볼린저 오버레이)
        ───────────+──────────────────────────────────────────
        거래량 레이블 | 볼륨 바
        ───────────+──────────────────────────────────────────
        RSI(14)   | RSI 라인 (* ^ v 문자)
        ───────────+──────────────────────────────────────────
        OBV{W}d   | Rolling OBV 바 (█ ░)
        [범례]
    """
    _df = _norm(df)
    if cols is not None:
        _df = _df.iloc[-cols:].reset_index(drop=True)

    n = len(_df)
    if n == 0:
        raise ValueError("DataFrame이 비어 있습니다.")

    o_arr = _df["open"].values.astype(float)
    h_arr = _df["high"].values.astype(float)
    l_arr = _df["low"].values.astype(float)
    c_arr = _df["close"].values.astype(float)

    min_p = float(l_arr.min())
    max_p = float(h_arr.max())

    # ── 1. 가격 그리드 ──────────────────────────────────────────────────────
    grid: list[list[str]] = [[EMPTY] * n for _ in range(rows)]
    for ci in range(n):
        col = [EMPTY] * rows
        _render_candle(col, o_arr[ci], h_arr[ci], l_arr[ci], c_arr[ci],
                       min_p, max_p, rows)
        for r in range(rows):
            grid[r][ci] = col[r]

    # ── 2. 지표 오버레이 (MA20/MA60/볼린저) ─────────────────────────────────
    indicators = {**add_ma(_df, [20, 60]), **add_bollinger(_df)}
    for name, series in indicators.items():
        ch  = _COMBINED_IND_CHARS.get(name, ".")
        arr = series.values.astype(float)
        for ci in range(min(n, len(arr))):
            v = arr[ci]
            if np.isnan(v):
                continue
            ri = _price_to_row(v, min_p, max_p, rows)
            if grid[ri][ci] not in _CANDLE_CHARS:
                grid[ri][ci] = ch

    # ── 3. 가격 축 레이블 폭 계산 ────────────────────────────────────────────
    label_row_set = {0, rows // 4, rows // 2, 3 * rows // 4, rows - 1}
    label_prices  = {r: _row_to_price(r, min_p, max_p, rows) for r in label_row_set}
    lw = max(len(_fmt_price(p)) for p in label_prices.values())

    # ── 4. 가격 차트 렌더링 ───────────────────────────────────────────────────
    lines: list[str] = []
    for ri in range(rows):
        row_str = "".join(grid[ri])
        lbl = (_fmt_price(label_prices[ri]).rjust(lw)
               if ri in label_row_set else " " * lw)
        lines.append(f"{lbl} |{row_str}")

    lines.append(" " * lw + "-+" + "-" * n)

    # ── 5. 거래량 바 ─────────────────────────────────────────────────────────
    if vol_rows > 0 and "volume" in _df.columns:
        v_arr  = _df["volume"].values.astype(float)
        max_v  = v_arr.max()
        vgrid  = [[" "] * n for _ in range(vol_rows)]
        for ci in range(n):
            if v_arr[ci] <= 0 or max_v == 0:
                continue
            bar  = max(1, round(v_arr[ci] / max_v * vol_rows))
            v_ch = VOL_BULL if c_arr[ci] >= o_arr[ci] else VOL_BEAR
            for r in range(vol_rows - bar, vol_rows):
                vgrid[r][ci] = v_ch
        vol_lbl = _fmt_vol(max_v)
        for ri, vrow in enumerate(vgrid):
            lbl = vol_lbl.rjust(lw) if ri == 0 else " " * lw
            lines.append(f"{lbl} |{''.join(vrow)}")

    # ── 6. RSI 서브차트 ───────────────────────────────────────────────────────
    if rsi_rows > 0:
        lines.extend(_render_rsi_subplot(_df, lw, n, rsi_rows, rsi_window))

    # ── 7. Rolling OBV 서브차트 ──────────────────────────────────────────────
    if obv_rows > 0:
        lines.extend(_render_obv_subplot(_df, lw, n, obv_rows, obv_window))

    # ── 8. 범례 ───────────────────────────────────────────────────────────────
    lines.append("")
    lines.append(f"  캔들: {BULL_BODY}=양봉  {BEAR_BODY}=음봉  {DOJI_BODY}=도지  {WICK}=꼬리")
    lines.append("  지표: [.]MA20  [-]MA60  [^]BB상단  [·]BB중단  [v]BB하단")
    lines.append(f"  RSI:  [{RSI_DOT}]중립  [{RSI_OB}]과매수(>70)  [{RSI_OS}]과매도(<30)  [─]30/70선  [:]50선")
    lines.append(f"  OBV:  [{OBV_BULL}]매수압력  [{OBV_BEAR}]매도압력  [─]중립선  ({obv_window}일 롤링)")

    chart_str = "\n".join(lines)

    # ── JSON 저장 ─────────────────────────────────────────────────────────────
    if save_path is not None:
        import json, datetime
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "chart": chart_str,
            "params": {
                "rows": rows, "cols": cols,
                "vol_rows": vol_rows,
                "rsi_rows": rsi_rows, "rsi_window": rsi_window,
                "obv_rows": obv_rows, "obv_window": obv_window,
            },
            "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            **(save_meta or {}),
        }
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    return chart_str


# ══════════════════════════════════════════════════════════════════════════════
#  데모
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os

    try:
        import yfinance as yf
    except ImportError:
        raise SystemExit("pip install yfinance  을 먼저 실행하세요.")

    # ── 설정 ──────────────────────────────────────────────────────────────────
    TICKER_FILE = "ticker.txt"   # 티커 목록 파일 (한 줄에 하나)
    DATA_DIR    = "data"         # 결과 저장 폴더
    PERIOD      = "6mo"
    ROWS        = 25
    COLS        = 80
    VOL_ROWS    = 5

    IND_CHARS = {
        "MA20": ".", "MA60": "-",
        "BB_upper": "^", "BB_mid": "\u00b7", "BB_lower": "v",
    }

    # ── ticker.txt 읽기 ───────────────────────────────────────────────────────
    if not os.path.exists(TICKER_FILE):
        raise SystemExit(
            f"'{TICKER_FILE}' \ud30c\uc77c\uc774 \uc5c6\uc2b5\ub2c8\ub2e4.\n"
            "\ud55c \uc904\uc5d0 \ud2f0\ucee4 \ud558\ub098\uc529 \uc791\uc131\ud558\uc138\uc694.  \uc608:\n"
            "  005930.KS\n  000660.KS\n  AAPL"
        )

    with open(TICKER_FILE, encoding="utf-8") as f:
        tickers = [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        ]

    if not tickers:
        raise SystemExit(f"'{TICKER_FILE}' \uc5d0 \uc720\ud6a8\ud55c \ud2f0\ucee4\uac00 \uc5c6\uc2b5\ub2c8\ub2e4.")

    print(f"\uc804\uccb4 {len(tickers)}\uac1c \ud2f0\ucee4 \ucc98\ub9ac \uc2dc\uc791  \u2192  \uc800\uc7a5 \ud3f4\ub354: {DATA_DIR}/\n")

    # ── data 폴더 생성 ────────────────────────────────────────────────────────
    os.makedirs(DATA_DIR, exist_ok=True)

    # ── 티커별 처리 ───────────────────────────────────────────────────────────
    ok_list:   list[str] = []
    fail_list: list[str] = []

    for ticker in tickers:
        print(f"  [{ticker}] \ub2e4\uc6b4\ub85c\ub4dc \uc911...", end=" ", flush=True)

        try:
            raw = yf.download(ticker, period=PERIOD,
                              progress=False, auto_adjust=True)
        except Exception as e:
            print(f"\uc2e4\ud328 (\ub2e4\uc6b4\ub85c\ub4dc \uc624\ub958: {e})")
            fail_list.append(ticker)
            continue

        if len(raw) == 0:
            print("\uc2e4\ud328 (\ube48 \ub370\uc774\ud130)")
            fail_list.append(ticker)
            continue

        # 컬럼 정규화
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw.columns = raw.columns.str.lower()

        try:
            # 지표 생성
            inds = {
                **add_ma(raw, [20, 60]),
                **add_bollinger(raw, window=20),
            }

            # 차트 텍스트 생성
            chart_text = plot_text_chart(
                raw,
                rows=ROWS, cols=COLS, vol_rows=VOL_ROWS,
                indicators=inds, indicator_chars=IND_CHARS,
                show_meta=True,
            )

            # 파일명: 티커의 특수문자 (./) → _ 치환
            safe_name = ticker.replace(".", "_").replace("/", "_")
            out_path  = os.path.join(DATA_DIR, f"{safe_name}.txt")

            sep = "\u2550" * 62
            header = (
                f"TICKER : {ticker}\n"
                f"PERIOD : {PERIOD}\n"
                f"ROWS   : {ROWS}  COLS: {COLS}\n"
                f"{sep}\n\n"
            )

            with open(out_path, "w", encoding="utf-8") as f:
                f.write(header + chart_text + "\n")

            print(f"\uc800\uc7a5 \uc644\ub8cc \u2192 {out_path}  ({len(raw)}\ubd09)")
            ok_list.append(ticker)

        except Exception as e:
            print(f"\uc2e4\ud328 (\ucc28\ud2b8 \uc0dd\uc131 \uc624\ub958: {e})")
            fail_list.append(ticker)

    # ── 결과 요약 ─────────────────────────────────────────────────────────────
    sep2 = "\u2500" * 50
    print(f"\n{sep2}")
    print(f"\uc644\ub8cc: {len(ok_list)}\uac1c  /  \uc2e4\ud328: {len(fail_list)}\uac1c")
    if fail_list:
        print("\uc2e4\ud328 \ud2f0\ucee4:", ", ".join(fail_list))
    print(f"\uc800\uc7a5 \uc704\uce58: {os.path.abspath(DATA_DIR)}/")
