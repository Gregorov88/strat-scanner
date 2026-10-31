"""Wyszukiwarka płynnych opcji — logika jak w ibquery.py, źródło: Yahoo Finance (opóźnione ~15 min).

Kolumny i filtry jak w skrypcie: ITM (domyślnie) / OTM, strike w zakresie ±range% od ceny,
OI ≥ min_oi, wolumen ≥ min_vol, delta ± tolerancja, sortowanie Ext/Int (ITM) lub Ext/Price (OTM) malejąco,
zielony wiersz gdy OI > 250 i Ext/Price < 30%.
Yahoo nie podaje greków — delta liczona z Blacka-Scholesa na zmienności implikowanej wyliczonej z MID.
"""
from __future__ import annotations

import math
from datetime import datetime, date
from zoneinfo import ZoneInfo

R = 0.04
NY = ZoneInfo("America/New_York")


def _ncdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_price(S, K, T, s, right):
    if T <= 0 or s <= 0:
        return max(0.0, S - K) if right == "C" else max(0.0, K - S)
    d1 = (math.log(S / K) + (R + s * s / 2) * T) / (s * math.sqrt(T))
    d2 = d1 - s * math.sqrt(T)
    if right == "C":
        return S * _ncdf(d1) - K * math.exp(-R * T) * _ncdf(d2)
    return K * math.exp(-R * T) * _ncdf(-d2) - S * _ncdf(-d1)


def bs_delta(S, K, T, s, right):
    if T <= 0 or s <= 0:
        itm = S > K if right == "C" else S < K
        return (1.0 if itm else 0.0) * (1 if right == "C" else -1)
    d1 = (math.log(S / K) + (R + s * s / 2) * T) / (s * math.sqrt(T))
    return _ncdf(d1) if right == "C" else _ncdf(d1) - 1


def implied_vol(price, S, K, T, right):
    intrinsic = max(0.0, S - K * math.exp(-R * T)) if right == "C" else max(0.0, K * math.exp(-R * T) - S)
    if price <= intrinsic + 1e-6 or T <= 0:
        return None
    lo, hi = 0.01, 5.0
    if bs_price(S, K, T, hi, right) < price:
        return None
    for _ in range(70):
        mid = (lo + hi) / 2
        if bs_price(S, K, T, mid, right) > price:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def _num(x):
    try:
        x = float(x)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def metrics(rows, spot, right, find_otm, min_oi, min_vol, delta, tol):
    """Wspólne filtry i wskaźniki (jak find_liquid_options w ibquery.py)."""
    out, no_delta = [], 0
    for r in rows:
        if r["oi"] < min_oi or r["volume"] < min_vol:
            continue
        if delta is not None:
            if r["delta"] is None:
                no_delta += 1
                continue
            if not (delta - tol <= abs(r["delta"]) <= delta + tol):
                continue
        K, P = r["strike"], r["last"]
        intr = max(0.0, spot - K) if right == "C" else max(0.0, K - spot)
        ext = max(0.0, P - intr)
        r = {**r, "intrinsic": round(intr, 2), "extrinsic": round(ext, 2),
             "ext_int": round(ext / intr * 100, 1) if intr > 0 else 0.0,
             "ext_price": round(ext / P * 100, 1) if P > 0 else 0.0}
        r["green"] = bool(r["oi"] > 250 and r["ext_price"] < 30)
        out.append(r)
    key = "ext_price" if find_otm else "ext_int"
    out.sort(key=lambda x: x[key], reverse=True)
    return out, no_delta


def pick_best(rows, delta=None):
    """Propozycja: zielony wiersz, najwęższy spread, potem najniższe Ext/Price, delta najbliżej celu."""
    if not rows:
        return None
    pool = [r for r in rows if r["green"] and r["bid"] is not None] or [r for r in rows if r["bid"] is not None] or rows
    def k(r):
        return ((r["spread_pct"] if r["spread_pct"] is not None else 99) // 2,      # spread w koszykach co 2%
                r["ext_price"], abs(abs(r["delta"] or 0) - (delta or 0.7)))
    return sorted(pool, key=k)[0]


def yahoo_chain(ticker: str, right: str = "C", delta: float | None = 0.7, tol: float = 0.05,
                min_dte: int = 5, max_dte: int = 31, min_oi: int = 0, min_vol: int = 10,
                range_pct: float = 30.0, find_otm: bool = False) -> dict:
    import yfinance as yf
    tk = yf.Ticker(ticker.replace(".", "-"))
    try:
        spot = float(tk.fast_info["lastPrice"])
    except Exception:
        spot = float(tk.history(period="5d")["Close"].iloc[-1])
    now = datetime.now(NY)
    today = now.date()
    exps = []
    for e in tk.options:
        dte = (date.fromisoformat(e) - today).days
        if min_dte <= dte <= max_dte:
            exps.append((e, dte))
    lo, hi = spot * (1 - range_pct / 100), spot * (1 + range_pct / 100)
    raw = []
    for e, dte in exps:
        ch = tk.option_chain(e)
        df = ch.calls if right == "C" else ch.puts
        T = max(dte - (now.hour + now.minute / 60) / 24 + 16 / 24, 0.5) / 365
        for _, x in df.iterrows():
            K = float(x["strike"])
            if not (lo <= K <= hi):
                continue
            itm = spot > K if right == "C" else spot < K
            if itm == find_otm:
                continue
            bid, ask = _num(x["bid"]), _num(x["ask"])
            if not (bid and ask and ask >= bid):
                bid = ask = None
            mid = (bid + ask) / 2 if bid else None
            vol = int(_num(x["volume"]) or 0)
            lastp = _num(x["lastPrice"]) or 0.0
            last = lastp if (vol > 0 and lastp > 0) else (mid or lastp)
            ref = mid or last
            iv = implied_vol(ref, spot, K, T, right) if ref else None
            if iv is None:
                y_iv = _num(x["impliedVolatility"])
                iv = y_iv if y_iv and 0.03 < y_iv < 4 else None
            dl = bs_delta(spot, K, T, iv, right) if iv else (1.0 if right == "C" else -1.0) if itm and ref and ref <= (abs(spot - K) * 1.01) else None
            raw.append({
                "contract": x["contractSymbol"], "expiration": e, "dte": dte, "strike": K,
                "last": round(float(last), 2), "bid": bid, "ask": ask,
                "spread": round(ask - bid, 2) if bid else None,
                "spread_pct": round((ask - bid) / mid * 100, 2) if bid else None,
                "quote": "DELAYED" if bid else "N/A", "volume": vol, "oi": int(_num(x["openInterest"]) or 0),
                "delta": round(dl, 3) if dl is not None else None, "iv": round(iv * 100, 1) if iv else None,
            })
    rows, no_delta = metrics(raw, spot, right, find_otm, min_oi, min_vol, delta, tol)
    return {"ticker": ticker, "right": right, "spot": round(spot, 2), "source": "yahoo",
            "expirations": [e for e, _ in exps], "quoted": len(raw), "no_delta": no_delta,
            "rows": rows, "best": pick_best(rows, delta),
            "asof": datetime.now(ZoneInfo("Europe/Warsaw")).strftime("%Y-%m-%d %H:%M")}


def _get_json(url):
    import requests
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (StratScanner)"}, timeout=20, allow_redirects=True)
    r.raise_for_status()
    return r.json()


def cboe_chain(ticker: str, right: str = "C", delta: float | None = 0.7, tol: float = 0.05,
               min_dte: int = 5, max_dte: int = 31, min_oi: int = 0, min_vol: int = 10,
               range_pct: float = 30.0, find_otm: bool = False) -> dict:
    """Opcje z publicznego API Cboe (opóźnione ~15 min) — z prawdziwymi grekami (delta, IV)."""
    import re
    d = None
    for sym in dict.fromkeys([ticker.upper(), ticker.upper().replace(".", ""), ticker.upper().replace(".", "/")]):
        try:
            d = _get_json(f"https://cdn.cboe.com/api/global/delayed_quotes/options/{sym}.json")["data"]
            break
        except Exception:
            continue
    if not d:
        raise RuntimeError("Cboe: brak danych")
    spot = float(d.get("current_price") or d.get("close"))
    today = datetime.now(NY).date()
    lo, hi = spot * (1 - range_pct / 100), spot * (1 + range_pct / 100)
    raw, exps = [], set()
    for x in d.get("options", []):
        m = re.match(r"^(.+?)(\d{6})([CP])(\d{8})$", x.get("option", ""))
        if not m or m.group(3) != right:
            continue
        e = date(2000 + int(m.group(2)[:2]), int(m.group(2)[2:4]), int(m.group(2)[4:]))
        dte = (e - today).days
        if not (min_dte <= dte <= max_dte):
            continue
        exps.add(e.isoformat())
        K = int(m.group(4)) / 1000
        if not (lo <= K <= hi):
            continue
        itm = spot > K if right == "C" else spot < K
        if itm == find_otm:
            continue
        bid, ask = _num(x.get("bid")), _num(x.get("ask"))
        if not (bid and ask and ask >= bid):
            bid = ask = None
        mid = (bid + ask) / 2 if bid else None
        vol = int(_num(x.get("volume")) or 0)
        lastp = _num(x.get("last_trade_price")) or 0.0
        last = lastp if (vol > 0 and lastp > 0) else (mid or lastp)
        dl = _num(x.get("delta"))
        iv = _num(x.get("iv"))
        if not dl or not iv:          # brak greków — licz z Blacka-Scholesa
            T = max(dte + 0.5, 0.5) / 365
            ref = mid or last
            ivc = implied_vol(ref, spot, K, T, right) if ref else None
            if ivc:
                iv, dl = ivc, bs_delta(spot, K, T, ivc, right)
            elif not dl:
                dl = None
        raw.append({
            "contract": x["option"], "expiration": e.isoformat(), "dte": dte, "strike": K,
            "last": round(float(last), 2), "bid": bid, "ask": ask,
            "spread": round(ask - bid, 2) if bid else None,
            "spread_pct": round((ask - bid) / mid * 100, 2) if bid else None,
            "quote": "DELAYED" if bid else "N/A", "volume": vol, "oi": int(_num(x.get("open_interest")) or 0),
            "delta": round(dl, 3) if dl is not None else None, "iv": round(iv * 100, 1) if iv else None,
        })
    rows, no_delta = metrics(raw, spot, right, find_otm, min_oi, min_vol, delta, tol)
    return {"ticker": ticker, "right": right, "spot": round(spot, 2), "source": "cboe",
            "expirations": sorted(exps), "quoted": len(raw), "no_delta": no_delta,
            "rows": rows, "best": pick_best(rows, delta),
            "asof": datetime.now(ZoneInfo("Europe/Warsaw")).strftime("%Y-%m-%d %H:%M")}


def delayed_chain(ticker: str, **kw) -> dict:
    """Cboe (z grekami) jako pierwsze źródło, Yahoo jako zapas."""
    try:
        r = cboe_chain(ticker, **kw)
        if r["quoted"]:
            return r
    except Exception:
        r = None
    try:
        y = yahoo_chain(ticker, **kw)
        if y["quoted"] or r is None:
            return y
    except Exception:
        if r is None:
            raise
    return r


if __name__ == "__main__":
    import json, sys
    r = delayed_chain(sys.argv[1] if len(sys.argv) > 1 else "AAPL", right=sys.argv[2] if len(sys.argv) > 2 else "C")
    print(json.dumps({k: v for k, v in r.items() if k != "rows"}, indent=1))
    for x in r["rows"][:15]:
        print(x)
