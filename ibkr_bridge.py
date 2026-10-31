#!/usr/bin/env python3
"""Most IBKR dla Strat Scannera — uruchamiasz na swoim komputerze razem z TWS.

Aplikacja w przeglądarce pyta ten most (http://127.0.0.1:8765) o łańcuch opcji z TWS.
Logika pobierania jest taka sama jak w ibquery.py: połączenie tylko do odczytu (readonly — nie może składać zleceń),
strike'i ±range% od ceny, notowania w paczkach, OI z generic tick 101, delta modelowa IBKR,
fallback na dane „frozen”, gdy rynek opcji jest zamknięty.

Wymagania:  pip install ib_insync
TWS:        Configure → API → Settings → Enable ActiveX and Socket Clients, port 7497, Trusted IP 127.0.0.1
Start:      python ibkr_bridge.py           (albo dwuklik na start_ibkr_bridge.bat)
"""
import asyncio
import json
import logging
import math
import os
import time
from copy import copy
from datetime import datetime, date, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

asyncio.set_event_loop(asyncio.new_event_loop())
from ib_insync import IB, Stock, Option  # noqa: E402

IB_HOST = os.getenv("IBQUERY_HOST", "127.0.0.1")
IB_PORT = int(os.getenv("IBQUERY_PORT", "7497"))
IB_CLIENT_ID = int(os.getenv("IBQUERY_CLIENT_ID", "144"))
BRIDGE_PORT = int(os.getenv("BRIDGE_PORT", "8765"))
BATCH = 45
BATCH_WAIT = 6.0
logging.getLogger("ib_insync").setLevel(logging.CRITICAL)

ib = IB()


def _ok(x):
    return x is not None and not (isinstance(x, float) and math.isnan(x)) and x > 0


def ensure():
    if not ib.isConnected():
        ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID, timeout=15, readonly=True)
    ib.reqMarketDataType(1)


def quote_metrics(bid, ask):
    valid = all(isinstance(x, (int, float)) and math.isfinite(x) and x > 0 for x in (bid, ask)) and ask >= bid
    if not valid:
        return None, None, None, None
    mid = (bid + ask) / 2
    return float(bid), float(ask), float(ask - bid), (ask - bid) / mid * 100


def stock_price(stock):
    t = ib.reqTickers(stock)[0]
    for p in (t.marketPrice(), t.last, t.close):
        if _ok(p):
            return float(p)
    raise ValueError(f"Brak ceny dla {stock.symbol}")


def contracts_for(stock, right, min_dte, max_dte, spot, find_otm, range_pct):
    chains = [c for c in ib.reqSecDefOptParams(stock.symbol, "", stock.secType, stock.conId) if c.exchange == "SMART"]
    if not chains:
        raise ValueError(f"Brak łańcucha opcji dla {stock.symbol}")
    chain = next((c for c in chains if c.tradingClass == stock.symbol), chains[0])
    today = date.today()
    wanted = []
    for e in sorted(chain.expirations):
        d = date(int(e[:4]), int(e[4:6]), int(e[6:8]))
        if min_dte <= (d - today).days <= max_dte:
            wanted.append(e)
    lo, hi = spot * (1 - range_pct / 100), spot * (1 + range_pct / 100)
    strikes = [k for k in sorted(chain.strikes)
               if lo <= k <= hi and ((spot > k) if right == "C" else (spot < k)) != find_otm]
    raw = [Option(stock.symbol, e, k, right, "SMART", tradingClass=chain.tradingClass) for e in wanted for k in strikes]
    out = []
    for i in range(0, len(raw), 200):
        out.extend(c for c in ib.qualifyContracts(*raw[i:i + 200]) if c.conId)
    return out, wanted


def _quote(batch, done):
    tickers = [ib.reqMktData(c, "101", False, False) for c in batch]
    t0 = time.time()
    while time.time() - t0 < BATCH_WAIT:
        ib.sleep(0.5)
        if all(done(t) for t in tickers):
            break
    for c in batch:
        ib.cancelMktData(c)
    return tickers


def snapshot(contracts, right):
    oi_of = (lambda t: t.callOpenInterest) if right == "C" else (lambda t: t.putOpenInterest)
    rows = []
    today = date.today()
    for i in range(0, len(contracts), BATCH):
        batch = contracts[i:i + BATCH]
        ib.reqMarketDataType(1)
        live = [copy(t) for t in _quote(batch, lambda t: not math.isnan(oi_of(t)) and t.modelGreeks is not None)]
        quotes = list(live)
        missing = [j for j, t in enumerate(live) if quote_metrics(t.bid, t.ask)[0] is None]
        if missing:
            try:
                ib.reqMarketDataType(2)
                frozen = _quote([batch[j] for j in missing], lambda t: quote_metrics(t.bid, t.ask)[0] is not None)
                for j, t in zip(missing, frozen):
                    quotes[j] = copy(t)
            finally:
                ib.reqMarketDataType(1)
        for c, t, f in zip(batch, live, quotes):
            g = t.modelGreeks or f.modelGreeks or t.lastGreeks
            bid, ask, spread, spread_pct = quote_metrics(f.bid, f.ask)
            mid = (bid + ask) / 2 if bid is not None else 0
            last = t.last if _ok(t.last) and _ok(t.volume) else mid
            e = c.lastTradeDateOrContractMonth
            ed = date(int(e[:4]), int(e[4:6]), int(e[6:8]))
            rows.append({
                "contract": c.localSymbol, "expiration": ed.isoformat(), "dte": (ed - today).days,
                "strike": c.strike, "last": round(float(last or 0), 2),
                "bid": bid, "ask": ask, "spread": round(spread, 2) if spread is not None else None,
                "spread_pct": round(spread_pct, 2) if spread_pct is not None else None,
                "quote": {1: "LIVE", 2: "FROZEN", 3: "DELAYED", 4: "DEL-FRZ"}.get(getattr(f, "marketDataType", None), "UNKNOWN") if bid is not None else "N/A",
                "volume": int(t.volume) if _ok(t.volume) else 0,
                "oi": int(oi_of(t)) if _ok(oi_of(t)) else 0,
                "delta": round(g.delta, 3) if g is not None and g.delta is not None else None,
                "iv": round(g.impliedVol * 100, 1) if g is not None and g.impliedVol else None,
            })
    return rows


def metrics(rows, spot, right, find_otm, min_oi, min_vol, delta, tol):
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
    out.sort(key=lambda x: x["ext_price" if find_otm else "ext_int"], reverse=True)
    return out, no_delta


def pick_best(rows, delta=None):
    if not rows:
        return None
    pool = [r for r in rows if r["green"] and r["bid"] is not None] or [r for r in rows if r["bid"] is not None] or rows
    return sorted(pool, key=lambda r: ((r["spread_pct"] if r["spread_pct"] is not None else 99) // 2,
                                       r["ext_price"], abs(abs(r["delta"] or 0) - (delta or 0.7))))[0]


def options(q):
    tk = q.get("ticker", "AAPL").upper().replace("-", " ").replace(".", " ")
    right = "P" if q.get("type", "C").upper().startswith("P") else "C"
    delta = float(q["delta"]) if q.get("delta") not in (None, "", "none") else None
    tol = float(q.get("tol", 0.05))
    min_dte, max_dte = int(q.get("min_dte", 5)), int(q.get("max_dte", 31))
    min_oi, min_vol = int(q.get("oi", 0)), int(q.get("vol", 10))
    range_pct = float(q.get("range", 20))
    find_otm = q.get("otm", "0") in ("1", "true")
    ensure()
    got = ib.qualifyContracts(Stock(tk, "SMART", "USD"))
    if not got or not got[0].conId:
        raise ValueError(f"Nieznany ticker {tk}")
    stock = got[0]
    spot = stock_price(stock)
    cons, exps = contracts_for(stock, right, min_dte, max_dte, spot, find_otm, range_pct)
    raw = snapshot(cons, right)
    rows, no_delta = metrics(raw, spot, right, find_otm, min_oi, min_vol, delta, tol)
    return {"ticker": tk, "right": right, "spot": round(spot, 2), "source": "ibkr",
            "expirations": [f"{e[:4]}-{e[4:6]}-{e[6:]}" for e in exps], "quoted": len(raw), "no_delta": no_delta,
            "rows": rows, "best": pick_best(rows, delta), "asof": datetime.now().strftime("%Y-%m-%d %H:%M")}


class H(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Private-Network", "true")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path == "/health":
                try:
                    ensure()
                    self._send(200, {"ok": True, "tws": True, "port": IB_PORT})
                except Exception as e:
                    self._send(200, {"ok": True, "tws": False, "error": f"Brak połączenia z TWS na porcie {IB_PORT}: {e}"})
            elif u.path == "/options":
                print(f"{datetime.now():%H:%M:%S}  opcje {q.get('ticker')} {q.get('type')} …", flush=True)
                r = options(q)
                print(f"           {len(r['rows'])} kontraktów spełnia kryteria (z {r['quoted']})", flush=True)
                self._send(200, r)
            else:
                self._send(404, {"error": "nieznana ścieżka"})
        except Exception as e:
            print("   błąd:", e, flush=True)
            self._send(500, {"error": str(e)})

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    print(f"Most IBKR działa: http://127.0.0.1:{BRIDGE_PORT}   (TWS {IB_HOST}:{IB_PORT}, clientId {IB_CLIENT_ID}, tylko odczyt)")
    print("Zostaw to okno otwarte. W aplikacji wybierz źródło opcji: IBKR. Zamknięcie: Ctrl+C.")
    try:
        ensure()
        print("Połączono z TWS.")
    except Exception as e:
        print("Uwaga: nie udało się połączyć z TWS —", e, "\nUruchom TWS i włącz API; most spróbuje ponownie przy pierwszym zapytaniu.")
    HTTPServer(("127.0.0.1", BRIDGE_PORT), H).serve_forever()
