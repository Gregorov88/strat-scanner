import gc
import warnings
warnings.filterwarnings("ignore")
import io
import pickle
import json
import os
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import scanner
import data_store
import backtest
import options as opt

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "last_scan.json")
BT_CACHE = os.path.join(HERE, "last_backtest.pkl")
WAW = ZoneInfo("Europe/Warsaw")
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

state = {"running": False, "progress": 0.0, "stage": "", "error": None, "started": None, "auto": False}
bt_state = {"running": False, "progress": 0.0, "error": None}
scan = json.load(open(CACHE)) if os.path.exists(CACHE) else None
if scan:
    for _r in scan.get("results", []):
        if "chart" in _r:
            _r["_detail"] = json.dumps({"chart": _r.pop("chart"), "fvg_list": _r.pop("fvg_list", {})})
def _load_bt():
    try:
        return pickle.load(open(BT_CACHE, "rb")) if os.path.exists(BT_CACHE) else None
    except Exception:
        return None


bt = _load_bt()
bt_agg_cache = {}
lock = threading.Lock()
AUTO = {"enabled": os.getenv("AUTO_REFRESH", "1") == "1", "every_min": 30, "last": None}


class ScanReq(BaseModel):
    universe: str = "all"
    ema_tf: str = "setup"


def _run_backtest(data):
    global bt
    bt_state.update(running=True, progress=0.0, error=None)
    try:
        r = backtest.run(data, progress_cb=lambda p: bt_state.update(progress=round(p, 3)))
        bt = r
        bt_agg_cache.clear()
        pickle.dump(r, open(BT_CACHE, "wb"))
        del data
        gc.collect()
    except Exception as e:
        bt_state["error"] = str(e)
    finally:
        bt_state.update(running=False, progress=1.0)


def worker(req: ScanReq, auto=False):
    global scan
    try:
        def cb(p, s):
            state["progress"], state["stage"] = round(p, 3), s
        uni = scanner.get_universe()
        tickers = sorted(uni)
        first = not data_store.STORE["data"] and not os.path.exists(data_store.PATH)
        cb(0.0, "Pobieranie pełnej historii (pierwszy raz ~5 min)" if first else "Aktualizacja notowań")
        data = data_store.get_data(tickers + scanner.CONTEXT_TICKERS, progress_cb=lambda p: cb(0.7 * p, "Pobieranie danych"))
        r = scanner.run_scan("all", {"ema_tf": req.ema_tf}, data=data, compact=True,
                             progress_cb=lambda p, s: cb(0.7 + 0.3 * max(0.0, (p - 0.5) * 2), s))
        r["auto"] = auto
        scan = None
        gc.collect()
        scan = r
        json.dump(r, open(CACHE, "w"))
        state["error"] = None
        # backtest liczony na świeżych danych raz na dzień (lub gdy go brak)
        today = datetime.now(WAW).strftime("%Y-%m-%d")
        del data
        gc.collect()
        if not bt or bt.get("v") != backtest.VERSION or not str(bt.get("generated_at", "")).startswith(today):
            ctxs = set(scanner.CONTEXT_TICKERS)
            data = {k: v for k, v in data_store.STORE["data"].items() if k not in ctxs}
            threading.Thread(target=_run_backtest, args=(data,), daemon=True).start()
    except Exception as e:
        state["error"] = str(e)
    finally:
        state.update(running=False, progress=1.0, stage="Gotowe", auto=False)
        AUTO["last"] = time.time()


def start_scan(req: ScanReq, auto=False):
    with lock:
        if state["running"]:
            return False
        state.update(running=True, progress=0.0, stage="Start", error=None, started=time.time(), auto=auto)
        threading.Thread(target=worker, args=(req, auto), daemon=True).start()
        return True


def auto_loop():
    """Automatyczne odświeżanie: po starcie serwera i co 30 min w czasie sesji USA (pn–pt 15:00–22:30 czasu PL)."""
    time.sleep(5)
    start_scan(ScanReq(), auto=True)
    while True:
        time.sleep(60)
        if not AUTO["enabled"]:
            continue
        now = datetime.now(WAW)
        in_session = now.weekday() < 5 and (15, 0) <= (now.hour, now.minute) <= (22, 30)
        due = AUTO["last"] is None or time.time() - AUTO["last"] >= AUTO["every_min"] * 60
        if in_session and due:
            start_scan(ScanReq(), auto=True)


threading.Thread(target=auto_loop, daemon=True).start()


@app.post("/api/scan")
def start(req: ScanReq):
    started = start_scan(req)
    return {"started": started, **state}


@app.get("/api/status")
def status():
    return {**state, "generated_at": scan["generated_at"] if scan else None,
            "data": data_store.info(), "auto_refresh": AUTO["enabled"],
            "backtest": {**bt_state, "generated_at": bt.get("generated_at") if bt else None}}


@app.post("/api/auto/{on}")
def auto(on: int):
    AUTO["enabled"] = bool(on)
    return {"auto_refresh": AUTO["enabled"]}


def lite(r):
    return {k: v for k, v in r.items() if k not in ("chart", "fvg_list", "_detail")}


@app.get("/api/results")
def results():
    if not scan:
        return {"generated_at": None, "results": []}
    return {**{k: v for k, v in scan.items() if k != "results"}, "results": [lite(r) for r in scan["results"]]}


@app.get("/api/ticker/{t}")
def ticker(t: str):
    if scan:
        for r in scan["results"]:
            if r["ticker"] == t.upper():
                if "_detail" in r:
                    return {**lite(r), **json.loads(r["_detail"])}
                return r
    raise HTTPException(404, "Brak spółki w ostatnim skanie")


@app.get("/api/options/{t}")
def options_ep(t: str, type: str = "C", delta: float | None = 0.7, tol: float = 0.05, min_dte: int = 5,
               max_dte: int = 31, oi: int = 0, vol: int = 10, range: float = 20.0, otm: int = 0):
    try:
        return opt.delayed_chain(t.upper(), right="P" if type.upper().startswith("P") else "C", delta=delta,
                                 tol=tol, min_dte=min_dte, max_dte=max_dte, min_oi=oi, min_vol=vol,
                                 range_pct=range, find_otm=bool(otm))
    except Exception as e:
        raise HTTPException(502, f"Opcje (Cboe/Yahoo): {e}")


@app.get("/api/backtest")
def backtest_ep(tickers: str = "", years: int = 0):
    if not bt:
        return {"ready": False, **bt_state}
    key = (tickers.upper(), years)
    if key not in bt_agg_cache:
        ts = {x.strip() for x in tickers.upper().replace(";", ",").split(",") if x.strip()} or None
        bt_agg_cache[key] = backtest.aggregate(bt, ts, years or None)
    return {"ready": True, **bt_agg_cache[key]}


@app.get("/api/export.csv")
def export():
    df = scanner.to_rows(scan) if scan else None
    buf = io.StringIO()
    if df is not None:
        df.to_csv(buf, index=False, sep=";", decimal=",")
    return StreamingResponse(iter(["\ufeff" + buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=strat_scan.csv"})


@app.get("/download/{name}")
def download(name: str):
    if name not in ("ibkr_bridge.py", "start_ibkr_bridge.bat"):
        raise HTTPException(404)
    return FileResponse(os.path.join(HERE, name), filename=name, media_type="application/octet-stream")


WEB = os.path.join(HERE, "web")


@app.get("/")
def index():
    return FileResponse(os.path.join(WEB, "index.html"))


if os.path.isdir(WEB):
    app.mount("/static", StaticFiles(directory=WEB), name="static")
