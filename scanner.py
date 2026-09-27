"""
Strat Scanner — Nasdaq 100 + S&P 500
Silnik analizy: The Strat (2D-2U, 2U-2U, 2U-1-2U, F2D / 2U-2D, 2D-2D, 2D-1-2D, F2U), FTFC (W/M/Q), EMA10/EMA20, FVG (W/M).

Może działać:
  1) jako moduł backendu aplikacji (server.py),
  2) samodzielnie, np. w Google Colab:
        !pip install yfinance lxml
        !python scanner.py            -> zapisuje strat_scan.csv
"""
from __future__ import annotations

import io
import json
import math
import os
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
NY = ZoneInfo("America/New_York")
UA = {"User-Agent": "Mozilla/5.0 (StratScanner)"}

# --------------------------------------------------------------------------------------
# Listy spółek
# --------------------------------------------------------------------------------------

def _fetch_wiki_lists() -> dict:
    sp = pd.read_html(io.StringIO(requests.get(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", headers=UA, timeout=20).text))[0]
    nd = pd.read_html(io.StringIO(requests.get(
        "https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies", headers=UA, timeout=20).text))[0]
    ind_col = [c for c in nd.columns if str(c).startswith("ICB Industry")]
    return {
        "sp500": [{"t": str(r["Symbol"]).replace(".", "-"), "n": r["Security"], "s": r["GICS Sector"]}
                  for _, r in sp.iterrows()],
        "ndx": [{"t": str(r["Ticker"]).replace(".", "-"), "n": r["Company"],
                 "s": r[ind_col[0]] if ind_col else ""} for _, r in nd.iterrows()],
    }


def get_universe(refresh: bool = False) -> dict:
    """Zwraca {ticker: {name, sector, indices:[...]}}. Najpierw próbuje Wikipedii, potem cache."""
    path = os.path.join(HERE, "tickers.json")
    lists = None
    if refresh or not os.path.exists(path):
        try:
            lists = _fetch_wiki_lists()
            if len(lists["sp500"]) > 400 and len(lists["ndx"]) > 90:
                json.dump(lists, open(path, "w"))
        except Exception:
            lists = None
    if lists is None:
        lists = json.load(open(path))
    uni: dict = {}
    for key, label in (("ndx", "NDX"), ("sp500", "SPX")):
        for r in lists[key]:
            d = uni.setdefault(r["t"], {"name": r["n"], "sector": r["s"], "indices": []})
            d["indices"].append(label)
    return uni


# --------------------------------------------------------------------------------------
# Dane
# --------------------------------------------------------------------------------------

def download_daily(tickers: list[str], years: int = 10, progress_cb=None, chunk: int = 60) -> dict:
    import yfinance as yf
    out = {}
    chunks = [tickers[i:i + chunk] for i in range(0, len(tickers), chunk)]
    for ci, ch in enumerate(chunks):
        for attempt in range(2):
            try:
                df = yf.download(ch, period=f"{years}y", interval="1d", group_by="ticker",
                                 auto_adjust=False, progress=False, threads=True)
                break
            except Exception:
                time.sleep(2)
                df = None
        if df is not None and not df.empty:
            for t in ch:
                try:
                    sub = df[t] if isinstance(df.columns, pd.MultiIndex) else df
                    sub = sub[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Open", "High", "Low", "Close"])
                    if len(sub) > 60:
                        sub.index = pd.to_datetime(sub.index).tz_localize(None)
                        out[t] = sub
                except Exception:
                    pass
        if progress_cb:
            progress_cb((ci + 1) / len(chunks))
    return out


RULE = {"D": None, "W": "W-FRI", "M": "ME", "Q": "QE"}


def resample(d: pd.DataFrame, tf: str) -> pd.DataFrame:
    if tf == "D":
        return d
    r = d.resample(RULE[tf]).agg({"Open": "first", "High": "max", "Low": "min",
                                  "Close": "last", "Volume": "sum"}).dropna(subset=["Open"])
    return r


# --------------------------------------------------------------------------------------
# Logika The Strat
# --------------------------------------------------------------------------------------

def strat_type(h, l, ph, pl) -> str:
    if h > ph and l < pl:
        return "3"
    if h > ph:
        return "2U"
    if l < pl:
        return "2D"
    return "1"


def candle_label(o, h, l, c, ph, pl) -> str:
    t = strat_type(h, l, ph, pl)
    if t == "2U" and c < o:
        return "F2U"
    if t == "2D" and c > o:
        return "F2D"
    return t


def period_complete(last_date: pd.Timestamp, tf: str, now_ny: datetime) -> bool:
    """Czy ostatnia świeca danego interwału jest już zamknięta.
    Etykiety po resample to koniec okresu (piątek / koniec miesiąca / koniec kwartału)."""
    today = pd.Timestamp(now_ny.date())
    after_close = now_ny.hour > 16 or (now_ny.hour == 16 and now_ny.minute >= 5)
    end = pd.Timestamp(last_date)
    if tf in ("M", "Q"):
        # jeśli koniec okresu wypada w weekend, zamknięcie następuje w ostatni piątek
        while end.weekday() >= 5:
            end -= pd.Timedelta(days=1)
    if today > end:
        return True
    return today == end and after_close


def find_pivot_tp(h, l, k, entry, direction, wing=2):
    """TP = najbliższy niewybity swing pivot.
    Long: pivot high (high wyższe od `wing` świec z każdej strony) nad wejściem, niewybity przez późniejsze świece.
    Short: analogicznie pivot low pod wejściem. Szukamy wstecz od świecy sygnałowej k;
    najnowszy niewybity pivot jest jednocześnie najbliższy cenowo."""
    n = len(h)
    if direction == "long":
        run_max = -np.inf          # najwyższe high po kandydacie (do końca danych)
        for i in range(n - 1, wing - 1, -1):
            if i + wing < n and i <= k - 1:
                if h[i] > max(h[i - wing:i]) and h[i] > max(h[i + 1:i + 1 + wing]) and h[i] > entry and h[i] > run_max:
                    return float(h[i]), i
            run_max = max(run_max, h[i])
    else:
        run_min = np.inf
        for i in range(n - 1, wing - 1, -1):
            if i + wing < n and i <= k - 1:
                if l[i] < min(l[i - wing:i]) and l[i] < min(l[i + 1:i + 1 + wing]) and l[i] < entry and l[i] < run_min:
                    return float(l[i]), i
            run_min = min(run_min, l[i])
    return None, None


# formacje: nazwa -> (kierunek, czy 3-świecowa, warunek na świecę sygnałową k)
PATTERNS = [
    ("2D-2U",   "long",  lambda lab, typ, k: lab(k) == "2D"),
    ("2U-2U",   "long",  lambda lab, typ, k: lab(k) == "2U"),
    ("2U-1-2U", "long",  lambda lab, typ, k: typ(k) == "1" and lab(k - 1) == "2U"),
    ("F2D",     "long",  lambda lab, typ, k: lab(k) == "F2D"),
    ("2U-2D",   "short", lambda lab, typ, k: lab(k) == "2U"),
    ("2D-2D",   "short", lambda lab, typ, k: lab(k) == "2D"),
    ("2D-1-2D", "short", lambda lab, typ, k: typ(k) == "1" and lab(k - 1) == "2D"),
    ("F2U",     "short", lambda lab, typ, k: lab(k) == "F2U"),
]

PAT_DESC = {
    "2D-2U": "Świeca sygnałowa 2D (czerwona). Wejście: przebicie jej high → kolejna świeca 2U (odwrócenie w górę).",
    "2U-2U": "Świeca sygnałowa 2U (zielona). Wejście: przebicie jej high → kolejna świeca 2U (kontynuacja wzrostu).",
    "2U-1-2U": "Świeca 2U (zielona) + inside bar (1). Wejście: przebicie high insidera → trzecia świeca 2U (kontynuacja).",
    "F2D": "Świeca F2D: wybiła low poprzedniej i zamknęła się na zielono. Wejście: przebicie jej high.",
    "2U-2D": "Świeca sygnałowa 2U (zielona). Wejście: przebicie jej low → kolejna świeca 2D (odwrócenie w dół).",
    "2D-2D": "Świeca sygnałowa 2D (czerwona). Wejście: przebicie jej low → kolejna świeca 2D (kontynuacja spadku).",
    "2D-1-2D": "Świeca 2D (czerwona) + inside bar (1). Wejście: przebicie low insidera → trzecia świeca 2D (kontynuacja).",
    "F2U": "Świeca F2U: wybiła high poprzedniej i zamknęła się na czerwono. Wejście: przebicie jej low.",
}


def detect_setups(bars: pd.DataFrame, tf: str, complete: bool) -> list[dict]:
    """Tylko setupy PRZED wejściem: świeca sygnałowa (dla 2-1-2 inside bar) jest ostatnią świecą,
    a trigger nie został jeszcze przebity.
      Oczekuje  – świeca sygnałowa zamknięta; bieżąca (jeśli już się formuje) nie wybiła ani high, ani low sygnałowej.
      W trakcie – świeca sygnałowa to bieżąca, jeszcze niezamknięta świeca (może się zmienić do zamknięcia).
    Wejście = high (long) / low (short) świecy sygnałowej, SL = przeciwna krawędź, TP = najbliższy niewybity swing pivot."""
    if len(bars) < 10:
        return []
    o, h, l, c = (bars[k].values for k in ("Open", "High", "Low", "Close"))
    i0 = len(bars) - 1
    lab = lambda i: candle_label(o[i], h[i], l[i], c[i], h[i - 1], l[i - 1])
    typ = lambda i: strat_type(h[i], l[i], h[i - 1], l[i - 1])
    last = float(c[i0])

    cands = []   # (index świecy sygnałowej, status)
    if complete:
        cands.append((i0, "Oczekuje"))
    else:
        cands.append((i0, "W trakcie"))
        if typ(i0) == "1":            # bieżąca nie wybiła jeszcze świecy sygnałowej z żadnej strony
            cands.append((i0 - 1, "Oczekuje"))

    res = []
    for k, status in cands:
        for name, direction, cond in PATTERNS:
            if k - 2 < 0 or not cond(lab, typ, k):
                continue
            entry = float(h[k]) if direction == "long" else float(l[k])
            sl = float(l[k]) if direction == "long" else float(h[k])
            tp, tp_i = find_pivot_tp(h, l, k, entry, direction)
            risk = abs(entry - sl)
            rr = abs(tp - entry) / risk if (tp is not None and risk > 0) else None
            res.append({
                "tf": tf, "pattern": name, "dir": direction, "status": status,
                "entry": round(entry, 2), "sl": round(sl, 2),
                "tp": round(tp, 2) if tp is not None else None,
                "tp_date": str(bars.index[tp_i].date()) if tp_i is not None else None,
                "rr": round(rr, 2) if rr is not None else None,
                "dist_entry_pct": round((entry - last) / last * 100, 2),
                "signal_date": str(bars.index[k].date()),
                "note": PAT_DESC[name] + (" Świeca sygnałowa jeszcze się formuje." if status == "W trakcie" else ""),
                "triggered": False, "tp_hit": False, "sl_hit": False, "rr_now": None,
            })
    return res


# --------------------------------------------------------------------------------------
# FVG
# --------------------------------------------------------------------------------------

def find_fvgs(bars: pd.DataFrame, lookback: int = 150) -> list[dict]:
    """Niewypełnione FVG (3-świecowe). Bycze: high[i-2] < low[i]. Niedźwiedzie: low[i-2] > high[i].
    FVG uznajemy za unieważnione, gdy świeca zamknie się poza jego dalszą krawędzią.
    Częściowe wypełnienie zawęża strefę."""
    b = bars.iloc[-lookback:]
    h, l, c = b["High"].values, b["Low"].values, b["Close"].values
    idx = b.index
    out = []
    n = len(b)
    for i in range(2, n):
        if h[i - 2] < l[i]:
            top, bot = l[i], h[i - 2]
            alive = True
            for j in range(i + 1, n):
                if c[j] < bot:
                    alive = False
                    break
                if l[j] < top:
                    top = max(bot, l[j])
            if alive and top > bot:
                out.append({"type": "bull", "top": float(top), "bot": float(bot), "date": str(idx[i - 1].date()),
                            "orig_top": float(l[i])})
        if l[i - 2] > h[i]:
            top, bot = l[i - 2], h[i]
            alive = True
            for j in range(i + 1, n):
                if c[j] > top:
                    alive = False
                    break
                if h[j] > bot:
                    bot = min(top, h[j])
            if alive and top > bot:
                out.append({"type": "bear", "top": float(top), "bot": float(bot), "date": str(idx[i - 1].date()),
                            "orig_bot": float(h[i])})
    return out


def fvg_context(bars: pd.DataFrame, price: float, near_pct: float) -> dict:
    """Najbliższe bycze FVG poniżej/wokół ceny oraz niedźwiedzie powyżej/wokół ceny."""
    fv = find_fvgs(bars)
    res = {"bull": None, "bear": None}
    for kind in ("bull", "bear"):
        best = None
        for f in fv:
            if f["type"] != kind:
                continue
            if f["bot"] <= price <= f["top"]:
                d, state = 0.0, "W FVG"
            elif kind == "bull" and price > f["top"]:
                d, state = (price - f["top"]) / price * 100, "Zbliża się"
            elif kind == "bear" and price < f["bot"]:
                d, state = (f["bot"] - price) / price * 100, "Zbliża się"
            else:
                continue
            if best is None or d < best["dist_pct"]:
                best = {"state": state, "dist_pct": round(d, 2), "top": round(f["top"], 2),
                        "bot": round(f["bot"], 2), "date": f["date"]}
        if best is not None:
            best["near"] = bool(best["dist_pct"] <= near_pct)
        res[kind] = best
    return res


# --------------------------------------------------------------------------------------
# Analiza jednej spółki
# --------------------------------------------------------------------------------------

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def analyze(ticker: str, d: pd.DataFrame, meta: dict, now_ny: datetime, cfg: dict) -> dict | None:
    price = float(d["Close"].iloc[-1])
    bars = {tf: resample(d, tf) for tf in ("D", "W", "M", "Q")}

    # FTFC: cena vs open bieżącej świecy D / W / M / Q (informacyjnie)
    ftfc = {}
    for tf in ("D", "W", "M", "Q"):
        op = float(bars[tf]["Open"].iloc[-1])
        ftfc[tf] = "up" if price > op else ("down" if price < op else "flat")
    if ftfc["D"] == ftfc["W"] == "up":
        ftfc_state = "FTFC Up"
    elif ftfc["D"] == ftfc["W"] == "down":
        ftfc_state = "FTFC Down"
    else:
        ftfc_state = "Konflikt"
    d_complete = period_complete(bars["D"].index[-1], "D", now_ny)
    w_complete = period_complete(bars["W"].index[-1], "W", now_ny)
    # otwarcie świecy, w której nastąpi wejście dla setupów oczekujących:
    # jeśli bieżąca świeca jest zamknięta, nowa otworzy się ~ na ostatnim zamknięciu
    d_open_next = price if d_complete else float(bars["D"]["Open"].iloc[-1])
    w_open_next = price if w_complete else float(bars["W"]["Open"].iloc[-1])
    d_open_now = float(bars["D"]["Open"].iloc[-1])
    w_open_now = float(bars["W"]["Open"].iloc[-1])

    # EMA10/EMA20 na D, W, M
    emas = {}
    for tf in ("D", "W", "M", "Q"):
        cl = bars[tf]["Close"]
        if len(cl) < 21:
            emas[tf] = None
            continue
        e10, e20 = float(ema(cl, 10).iloc[-1]), float(ema(cl, 20).iloc[-1])
        emas[tf] = {"e10": round(e10, 2), "e20": round(e20, 2),
                    "trend": "up" if e10 > e20 else "down",
                    "above": bool(price > e10 and price > e20), "below": bool(price < e10 and price < e20)}

    # FVG W i M
    fvg = {"W": fvg_context(bars["W"], price, cfg["near_w"]),
           "M": fvg_context(bars["M"], price, cfg["near_m"])}

    # Setupy na D (FVG W, FTFC D+W filtr, D+W+M ocena) lub W (FVG M, FTFC W+M filtr, W+M+Q ocena)
    RULES = {"D": {"fvg": "W", "ftfc": ("D", "W"), "ema": ("D", "W", "M")},
             "W": {"fvg": "M", "ftfc": ("W", "M"), "ema": ("W", "M", "Q")}}
    setups = []
    for stf in cfg["setup_tfs"]:
        rule = RULES[stf]
        b = bars[stf]
        comp = period_complete(b.index[-1], stf, now_ny)
        opens = {k: float(bars[k]["Open"].iloc[-1]) for k in rule["ftfc"]}
        for s in detect_setups(b, stf, comp):
            direction = s["dir"]
            sign = 1 if direction == "long" else -1
            want = "up" if direction == "long" else "down"
            # FTFC przy cenie triggera (wejście nastąpi na poziomie entry)
            fl = {k: bool(sign * (s["entry"] - opens[k]) > 0) for k in rule["ftfc"]}
            ftfc_ok = all(fl.values())
            ema_flags = {k: (None if not emas.get(k) else bool(emas[k]["trend"] == want)) for k in rule["ema"]}
            ema_ok = all(v is True for v in ema_flags.values())
            kind = "bull" if direction == "long" else "bear"
            fz = fvg[rule["fvg"]][kind]
            fvg_ok = bool(fz and fz["near"])
            n_ok = int(ftfc_ok) + int(ema_ok) + int(fvg_ok)
            if n_ok == 0:
                continue
            grade = {3: "A", 2: "B", 1: "C"}[n_ok]
            score = n_ok * 10 + (1 if fvg_ok and fz["state"] == "W FVG" else 0) + (1 if s["rr"] and s["rr"] >= 2 else 0)
            s.update({
                "ftfc_mode": "przy triggerze", "ftfc_tfs": list(rule["ftfc"]), "ftfc_flags": fl,
                "ftfc_ok": ftfc_ok, "ftfc_full": ftfc_ok,
                "gap_note": s["status"] == "Oczekuje" and comp,
                "ema_ok": ema_ok, "ema_tfs": list(rule["ema"]), "ema_flags": ema_flags,
                "fvg_ok": fvg_ok, "fvg_tfs": [rule["fvg"]] if fvg_ok else [],
                "fvg_state": fz["state"] if fvg_ok else None, "fvg_dist": fz["dist_pct"] if fvg_ok else None,
                "fvg_zone": [fz["bot"], fz["top"]] if fvg_ok else None,
                "score": score, "grade": grade, "n_ok": n_ok,
            })
            setups.append(s)

    # historia do mini-wykresu (60 świec dziennych / tygodniowych)
    def pack(b, n):
        e10s, e20s = ema(b["Close"], 10), ema(b["Close"], 20)
        b = b.iloc[-n:]
        return [[str(i.date()), round(float(r.Open), 2), round(float(r.High), 2), round(float(r.Low), 2),
                 round(float(r.Close), 2), round(float(e10s.loc[i]), 2), round(float(e20s.loc[i]), 2)]
                for i, r in b.iterrows()]

    def fvg_list(b, n=6):
        fv = find_fvgs(b)
        for f in fv:
            mid = (f["top"] + f["bot"]) / 2
            f["dist_pct"] = round(abs(mid - price) / price * 100, 2)
            f["top"], f["bot"] = round(f["top"], 2), round(f["bot"], 2)
        fv.sort(key=lambda f: f["dist_pct"])
        return fv[:n]

    last_types = {}
    for tf in ("D", "W", "M", "Q"):
        b = bars[tf]
        if len(b) >= 2:
            last_types[tf] = candle_label(*(float(b[k].iloc[-1]) for k in ("Open", "High", "Low", "Close")),
                                          float(b["High"].iloc[-2]), float(b["Low"].iloc[-2]))

    chg = (price / float(d["Close"].iloc[-2]) - 1) * 100 if len(d) > 1 else 0
    return {
        "ticker": ticker, "name": meta["name"], "sector": meta["sector"], "indices": meta["indices"],
        "price": round(price, 2), "chg": round(chg, 2), "last_date": str(d.index[-1].date()),
        "ftfc": ftfc, "ftfc_state": ftfc_state, "ema": emas, "fvg": fvg, "setups": setups,
        "candles": last_types,
        "chart": {"D": pack(bars["D"], 120), "W": pack(bars["W"], 104), "M": pack(bars["M"], 60)},
        "fvg_list": {"W": fvg_list(bars["W"]), "M": fvg_list(bars["M"])},
    }


DEFAULT_CFG = {"near_w": 3.0, "near_m": 3.0, "setup_tfs": ["D", "W"], "ema_tf": "setup"}


def run_scan(universe_filter: str = "all", cfg: dict | None = None, progress_cb=None, limit: int | None = None) -> dict:
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    uni = get_universe()
    tickers = [t for t, m in uni.items()
               if universe_filter == "all" or (universe_filter == "ndx" and "NDX" in m["indices"])
               or (universe_filter == "spx" and "SPX" in m["indices"])]
    tickers.sort()
    if limit:
        tickers = tickers[:limit]
    t0 = time.time()
    data = download_daily(tickers, progress_cb=(lambda p: progress_cb(0.85 * p, "Pobieranie danych")) if progress_cb else None)
    now_ny = datetime.now(NY)
    results, errors = [], []
    for k, t in enumerate(tickers):
        if t not in data:
            errors.append(t)
            continue
        try:
            r = analyze(t, data[t], uni[t], now_ny, cfg)
            if r:
                results.append(r)
        except Exception as e:  # noqa
            errors.append(t)
        if progress_cb and k % 25 == 0:
            progress_cb(0.85 + 0.15 * (k + 1) / len(tickers), "Analiza")
    return {"generated_at": datetime.now(ZoneInfo("Europe/Warsaw")).strftime("%Y-%m-%d %H:%M"),
            "data_date": max((r["last_date"] for r in results), default=None),
            "universe": universe_filter, "count": len(results), "errors": errors,
            "elapsed_s": round(time.time() - t0, 1), "cfg": cfg, "results": results}


def to_rows(scan: dict) -> pd.DataFrame:
    rows = []
    for r in scan["results"]:
        for s in r["setups"]:
            rows.append({
                "Ticker": r["ticker"], "Spółka": r["name"], "Indeks": "/".join(r["indices"]), "Cena": r["price"],
                "TF": s["tf"], "Setup": s["pattern"], "Kierunek": s["dir"], "Status": s["status"],
                "Ocena": s["grade"], "Wynik": s["score"], "Wejście": s["entry"], "SL": s["sl"], "TP": s["tp"],
                "R:R": s["rr"], "Do wejścia %": s["dist_entry_pct"], "FTFC pełne": "tak" if s["ftfc_full"] else "nie", "Tryb FTFC": s["ftfc_mode"],
                "D/W/M/Q": "/".join(r["ftfc"][k] for k in ("D", "W", "M", "Q")), "EMA OK": s["ema_ok"],
                "FVG": ",".join(s["fvg_tfs"]) or "-",
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["Wynik", "R:R"], ascending=[False, False])
    return df


if __name__ == "__main__":
    import sys
    uf = sys.argv[1] if len(sys.argv) > 1 else "all"
    scan = run_scan(uf, progress_cb=lambda p, s: print(f"\r{s}: {p*100:5.1f}%", end=""))
    print()
    df = to_rows(scan)
    df.to_csv("strat_scan.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(df[df["Ocena"].isin(["A", "B"])].head(60).to_string(index=False))
    print(f"\nZapisano strat_scan.csv — {len(df)} setupów, {scan['count']} spółek, {scan['elapsed_s']} s")
