"""
test_chart.py — text_chart.py 시각 확인용 테스트 스크립트

사용법:
    python test_chart.py                  # 기본 (첫 번째 pkl, 최근 42일)
    python test_chart.py 005930           # 특정 종목 코드
    python test_chart.py 005930 60        # 종목 코드 + 표시 기간(거래일)
    python test_chart.py 005930 60 --rows 25 --rsi 20 --obv 12
"""

import argparse
import glob
import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from text_chart import plot_combined_chart

# ── 인자 파싱 ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("code",    nargs="?", default=None,  help="종목 코드 (예: 005930)")
parser.add_argument("cols",    nargs="?", type=int, default=42, help="표시 거래일 수 (기본 42)")
parser.add_argument("--rows",  type=int, default=20,  help="가격 차트 높이 (기본 20)")
parser.add_argument("--vol",   type=int, default=4,   help="거래량 높이 (기본 4)")
parser.add_argument("--rsi",   type=int, default=15,  help="RSI 높이 (기본 15)")
parser.add_argument("--rsi-w", type=int, default=14,  help="RSI 기간 (기본 14)")
parser.add_argument("--obv",   type=int, default=10,  help="OBV 높이 (기본 10)")
parser.add_argument("--obv-w", type=int, default=20,  help="OBV 롤링 기간 (기본 20)")
parser.add_argument("--save",  action="store_true",   help="chart_save/ 에 JSON 저장")
args = parser.parse_args()

# ── pkl 로드 ──────────────────────────────────────────────────────────────────
cache_dir = Path(__file__).parent / "ohlcv_cache"

if args.code:
    code = args.code.zfill(6)
    pkl_path = cache_dir / f"{code}.pkl"
    if not pkl_path.exists():
        sys.exit(f"캐시 없음: {pkl_path}")
else:
    pkls = sorted(cache_dir.glob("*.pkl"))
    if not pkls:
        sys.exit("ohlcv_cache/ 에 pkl 파일이 없습니다.")
    pkl_path = pkls[0]
    code = pkl_path.stem

with open(pkl_path, "rb") as f:
    df = pickle.load(f)

# 컬럼 정규화
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
df.columns = df.columns.str.lower()
df = df.loc[:, ~df.columns.duplicated(keep="first")]
df = df.dropna(subset=["open", "high", "low", "close"]).sort_index()

# 표시 구간
cols = min(args.cols, len(df))
x_df = df.iloc[-cols:].copy()

print(f"종목: {code}  |  기간: {x_df.index[0].date()} ~ {x_df.index[-1].date()}  |  {cols}거래일")
print(f"설정: rows={args.rows}  vol={args.vol}  rsi={args.rsi}(w={args.rsi_w})  obv={args.obv}(w={args.obv_w})")
print()

save_path = None
if args.save:
    save_dir = Path(__file__).parent / "chart_save"
    save_dir.mkdir(exist_ok=True)
    save_path = str(save_dir / f"{code}_{x_df.index[-1].date()}.json")

chart = plot_combined_chart(
    x_df,
    rows=args.rows,
    cols=cols,
    vol_rows=args.vol,
    rsi_rows=args.rsi,
    rsi_window=args.rsi_w,
    obv_rows=args.obv,
    obv_window=args.obv_w,
    save_path=save_path,
    save_meta={"ticker": code, "x_start": str(x_df.index[0].date()),
               "x_end": str(x_df.index[-1].date())},
)
print(chart)
if save_path:
    print(f"\n저장: {save_path}")
