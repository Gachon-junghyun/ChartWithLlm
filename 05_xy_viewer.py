"""
05_xy_viewer.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
X/Y 페어 브라우저 뷰어 (로컬 웹서버, 외부 라이브러리 불필요)

실행: python 05_xy_viewer.py
→ 브라우저에서 http://localhost:8765 열림

기능
────
  · 종목 / 날짜 필터링
  · UP / DOWN / FLAT 방향별 필터
  · X 텍스트 차트 + 메타데이터 표시
  · Y 10일 결과 (등락률, 방향, 일별 그래프)
  · 키보드 ← → 로 이전/다음 페어 이동
"""

import json
import os
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# ── 경로 ──────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent
JSONL     = BASE_DIR / "xy_pairs" / "pairs.jsonl"
PORT      = 8765


# ── 데이터 로드 ───────────────────────────────────────────────────────────────

def load_pairs() -> list[dict]:
    if not JSONL.exists():
        return []
    with open(JSONL, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# ── HTML 렌더러 ────────────────────────────────────────────────────────────────

def render_y_bars(daily: list[dict]) -> str:
    """Y 10일 미니 바 차트 HTML"""
    if not daily:
        return ""
    max_abs = max(abs(d["ret_pct"]) for d in daily) or 1
    bars = []
    for d in daily:
        pct = d["ret_pct"]
        color = "#e74c3c" if pct < 0 else "#2ecc71"
        height = max(4, int(abs(pct) / max_abs * 60))
        sign = "+" if pct >= 0 else ""
        bars.append(
            f'<div class="bar-wrap" title="{d["date"]}: {sign}{pct}%">'
            f'  <div class="bar-label">{sign}{pct:.1f}%</div>'
            f'  <div class="bar" style="height:{height}px;background:{color}"></div>'
            f'  <div class="bar-day">D{d["day"]}</div>'
            f'</div>'
        )
    return '<div class="bar-chart">' + "".join(bars) + '</div>'


def render_pair_card(pair: dict, idx: int, total: int) -> str:
    direction = pair["y_direction"]
    dir_color = {"UP": "#2ecc71", "DOWN": "#e74c3c", "FLAT": "#f39c12"}.get(direction, "#aaa")
    dir_arrow = {"UP": "▲", "DOWN": "▼", "FLAT": "━"}.get(direction, "?")
    ret = pair["y_return_pct"]
    ret_sign = "+" if ret >= 0 else ""

    chart_html = pair["x_chart"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    meta_html  = pair["x_meta"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    bars_html  = render_y_bars(pair.get("y_daily", []))

    return f"""
<div class="pair-card">
  <div class="pair-header">
    <div class="pair-id">
      <span class="ticker">{pair['ticker']}</span>
      <span class="name">{pair['name']}</span>
    </div>
    <div class="nav-info">
      <span class="nav-count">{idx + 1} / {total}</span>
      <a href="?idx={max(0, idx-1)}&{_filter_qs()}" class="nav-btn" id="prev">◀ 이전</a>
      <a href="?idx={min(total-1, idx+1)}&{_filter_qs()}" class="nav-btn" id="next">다음 ▶</a>
    </div>
  </div>

  <div class="pair-meta-row">
    <div class="meta-item">📅 X: {pair['x_start']} ~ {pair['x_end']}</div>
    <div class="meta-item">📅 Y: {pair['y_start']} ~ {pair['y_end']}</div>
    <div class="meta-item direction" style="color:{dir_color}">
      {dir_arrow} {direction} &nbsp; {ret_sign}{ret:.2f}%
    </div>
    <div class="meta-item">⬆ 최대 {pair['y_max_up']:+.2f}%</div>
    <div class="meta-item">⬇ 최대 {pair['y_max_down']:+.2f}%</div>
  </div>

  <div class="two-col">
    <div class="col-x">
      <div class="col-title">X — 6개월 텍스트 차트</div>
      <pre class="chart-pre">{chart_html}</pre>
      <div class="col-title" style="margin-top:12px">메타데이터</div>
      <pre class="meta-pre">{meta_html}</pre>
    </div>
    <div class="col-y">
      <div class="col-title">Y — 이후 10 거래일</div>
      {bars_html}
      <table class="y-table">
        <thead><tr><th>일차</th><th>날짜</th><th>시가</th><th>고가</th><th>저가</th><th>종가</th><th>등락률</th></tr></thead>
        <tbody>
          {''.join(
            f'<tr class="{"up" if d["ret_pct"]>=0 else "dn"}">'
            f'<td>D{d["day"]}</td><td>{d["date"]}</td>'
            f'<td>{d["open"]:,.0f}</td><td>{d["high"]:,.0f}</td>'
            f'<td>{d["low"]:,.0f}</td><td>{d["close"]:,.0f}</td>'
            f'<td>{"+" if d["ret_pct"]>=0 else ""}{d["ret_pct"]:.2f}%</td></tr>'
            for d in pair.get("y_daily", [])
          )}
        </tbody>
      </table>
    </div>
  </div>
</div>
"""


# 전역 필터 상태 (간단 구현)
_current_filter = {}

def _filter_qs() -> str:
    parts = []
    for k, v in _current_filter.items():
        if v:
            parts.append(f"{k}={v}")
    return "&".join(parts)


def render_page(pairs: list[dict], idx: int,
                ticker_filter: str, dir_filter: str, name_filter: str) -> str:

    filtered = pairs
    if ticker_filter:
        filtered = [p for p in filtered if ticker_filter.upper() in p["ticker"]]
    if name_filter:
        filtered = [p for p in filtered if name_filter in p["name"]]
    if dir_filter and dir_filter != "ALL":
        filtered = [p for p in filtered if p["y_direction"] == dir_filter]

    total = len(filtered)
    idx   = max(0, min(idx, total - 1))

    # 통계
    if pairs:
        all_up   = sum(1 for p in pairs if p["y_direction"] == "UP")
        all_dn   = sum(1 for p in pairs if p["y_direction"] == "DOWN")
        all_fl   = sum(1 for p in pairs if p["y_direction"] == "FLAT")
    else:
        all_up = all_dn = all_fl = 0

    card_html = render_pair_card(filtered[idx], idx, total) if filtered else "<div class='empty'>필터 결과 없음</div>"

    tickers_set = sorted(set(p["ticker"] for p in pairs))
    ticker_opts = "".join(
        f'<option value="{t}" {"selected" if t == ticker_filter else ""}>{t}</option>'
        for t in tickers_set
    )

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>KOSPI 200 X/Y 뷰어</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ background: #0d1117; color: #c9d1d9; font-family: 'Consolas', 'Monaco', monospace; font-size: 13px; }}
.top-bar {{ background: #161b22; border-bottom: 1px solid #30363d; padding: 10px 18px; display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }}
.top-bar h1 {{ font-size: 15px; color: #58a6ff; margin-right: 8px; white-space: nowrap; }}
.stat-badge {{ background: #21262d; border: 1px solid #30363d; border-radius: 4px; padding: 3px 10px; font-size: 12px; white-space: nowrap; }}
.stat-badge.up {{ color: #2ecc71; }} .stat-badge.dn {{ color: #e74c3c; }} .stat-badge.fl {{ color: #f39c12; }}
.filter-bar {{ background: #161b22; padding: 8px 18px; display: flex; align-items: center; gap: 10px; border-bottom: 1px solid #30363d; flex-wrap: wrap; }}
.filter-bar label {{ font-size: 12px; color: #8b949e; }}
select, input[type=text] {{ background: #21262d; border: 1px solid #30363d; color: #c9d1d9; padding: 4px 8px; border-radius: 4px; font-size: 12px; }}
.filter-btn {{ background: #238636; border: none; color: white; padding: 4px 14px; border-radius: 4px; cursor: pointer; font-size: 12px; }}
.filter-btn:hover {{ background: #2ea043; }}
.content {{ padding: 14px 18px; }}
.pair-card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 14px; }}
.pair-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }}
.ticker {{ font-size: 18px; font-weight: bold; color: #58a6ff; }}
.name {{ font-size: 14px; color: #8b949e; margin-left: 8px; }}
.nav-info {{ display: flex; align-items: center; gap: 8px; }}
.nav-count {{ color: #8b949e; font-size: 12px; }}
.nav-btn {{ background: #21262d; border: 1px solid #30363d; color: #c9d1d9; padding: 4px 12px; border-radius: 4px; text-decoration: none; font-size: 12px; }}
.nav-btn:hover {{ background: #30363d; }}
.pair-meta-row {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 12px; padding: 8px 12px; background: #0d1117; border-radius: 6px; }}
.meta-item {{ font-size: 12px; color: #8b949e; white-space: nowrap; }}
.meta-item.direction {{ font-size: 15px; font-weight: bold; }}
.two-col {{ display: grid; grid-template-columns: 1fr 400px; gap: 14px; }}
@media (max-width: 1100px) {{ .two-col {{ grid-template-columns: 1fr; }} }}
.col-title {{ font-size: 11px; color: #58a6ff; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; border-bottom: 1px solid #21262d; padding-bottom: 4px; }}
.chart-pre {{ font-family: 'Consolas','Monaco','Courier New',monospace; font-size: 11px; line-height: 1.3; background: #0d1117; padding: 10px; border-radius: 6px; overflow-x: auto; white-space: pre; color: #c9d1d9; border: 1px solid #21262d; }}
.meta-pre {{ font-size: 11px; background: #0d1117; padding: 8px 10px; border-radius: 6px; line-height: 1.6; border: 1px solid #21262d; white-space: pre-wrap; }}
.bar-chart {{ display: flex; align-items: flex-end; gap: 4px; height: 100px; padding: 8px; background: #0d1117; border-radius: 6px; margin-bottom: 12px; border: 1px solid #21262d; }}
.bar-wrap {{ display: flex; flex-direction: column; align-items: center; flex: 1; }}
.bar {{ width: 100%; min-height: 4px; border-radius: 2px 2px 0 0; }}
.bar-label {{ font-size: 9px; color: #8b949e; margin-bottom: 2px; white-space: nowrap; }}
.bar-day {{ font-size: 9px; color: #8b949e; margin-top: 3px; }}
.y-table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
.y-table th {{ background: #21262d; padding: 5px 8px; text-align: right; color: #8b949e; font-weight: normal; }}
.y-table th:first-child, .y-table th:nth-child(2) {{ text-align: center; }}
.y-table td {{ padding: 4px 8px; border-bottom: 1px solid #21262d; text-align: right; }}
.y-table td:first-child, .y-table td:nth-child(2) {{ text-align: center; }}
.y-table tr.up td:last-child {{ color: #2ecc71; }}
.y-table tr.dn td:last-child {{ color: #e74c3c; }}
.empty {{ text-align: center; padding: 40px; color: #8b949e; }}
</style>
</head>
<body>

<div class="top-bar">
  <h1>📊 KOSPI 200 X/Y 뷰어</h1>
  <span class="stat-badge">전체 {len(pairs)}개 페어</span>
  <span class="stat-badge up">▲ UP {all_up}</span>
  <span class="stat-badge dn">▼ DOWN {all_dn}</span>
  <span class="stat-badge fl">━ FLAT {all_fl}</span>
  <span class="stat-badge" style="margin-left:auto">필터 결과: {total}개</span>
</div>

<form class="filter-bar" method="get" action="/">
  <label>종목코드</label>
  <select name="ticker">
    <option value="">전체</option>
    {ticker_opts}
  </select>
  <label>종목명</label>
  <input type="text" name="name" value="{name_filter}" placeholder="삼성" style="width:80px">
  <label>방향</label>
  <select name="dir">
    <option value="ALL" {"selected" if dir_filter in ("ALL","") else ""}>전체</option>
    <option value="UP"   {"selected" if dir_filter == "UP" else ""}>▲ UP</option>
    <option value="DOWN" {"selected" if dir_filter == "DOWN" else ""}>▼ DOWN</option>
    <option value="FLAT" {"selected" if dir_filter == "FLAT" else ""}>━ FLAT</option>
  </select>
  <input type="hidden" name="idx" value="0">
  <button class="filter-btn" type="submit">필터 적용</button>
</form>

<div class="content">
  {card_html}
</div>

<script>
document.addEventListener('keydown', function(e) {{
  const total = {total};
  const idx   = {idx};
  if (e.key === 'ArrowRight' && idx < total - 1) {{
    window.location.href = document.getElementById('next').href;
  }} else if (e.key === 'ArrowLeft' && idx > 0) {{
    window.location.href = document.getElementById('prev').href;
  }}
}});
</script>
</body>
</html>"""


# ── HTTP 핸들러 ────────────────────────────────────────────────────────────────

_pairs_cache: list[dict] = []

class Handler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        pass  # 콘솔 로그 조용히

    def do_GET(self):
        global _pairs_cache
        parsed = urlparse(self.path)
        qs     = parse_qs(parsed.query)

        idx          = int(qs.get("idx",    ["0"])[0])
        ticker_flt   = qs.get("ticker", [""])[0]
        dir_flt      = qs.get("dir",    ["ALL"])[0]
        name_flt     = qs.get("name",   [""])[0]

        html = render_page(_pairs_cache, idx, ticker_flt, dir_flt, name_flt).encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html)))
        self.end_headers()
        self.wfile.write(html)


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    global _pairs_cache
    _pairs_cache = load_pairs()

    if not _pairs_cache:
        print(f"[오류] {JSONL} 없음")
        print("먼저 01_download_ohlcv.py → 02_generate_xy.py 실행하세요.")
        sys.exit(1)

    total = len(_pairs_cache)
    up    = sum(1 for p in _pairs_cache if p["y_direction"] == "UP")
    dn    = sum(1 for p in _pairs_cache if p["y_direction"] == "DOWN")
    fl    = sum(1 for p in _pairs_cache if p["y_direction"] == "FLAT")

    print(f"페어 로드: {total}개  (UP {up} / DOWN {dn} / FLAT {fl})")
    print(f"브라우저 열기: http://localhost:{PORT}")
    print("종료: Ctrl+C")
    print()

    url = f"http://localhost:{PORT}"
    webbrowser.open(url)

    server = HTTPServer(("localhost", PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n종료.")


if __name__ == "__main__":
    main()
