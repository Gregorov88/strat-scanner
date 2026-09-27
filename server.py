import csv, io, json, os, threading, time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import scanner

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "last_scan.json")
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

state = {"running": False, "progress": 0.0, "stage": "", "error": None, "started": None}
scan = json.load(open(CACHE)) if os.path.exists(CACHE) else None
lock = threading.Lock()


class ScanReq(BaseModel):
    universe: str = "all"
    near_w: float = 3.0
    near_m: float = 3.0
    setup_tfs: list[str] = ["D", "W"]
    ema_tf: str = "setup"


def worker(req: ScanReq):
    global scan
    try:
        def cb(p, s):
            state["progress"], state["stage"] = round(p, 3), s
        r = scanner.run_scan(req.universe, {"near_w": 3.0, "near_m": 3.0,
                                            "setup_tfs": ["D", "W"], "ema_tf": req.ema_tf}, progress_cb=cb)
        scan = r
        json.dump(r, open(CACHE, "w"))
        state["error"] = None
    except Exception as e:
        state["error"] = str(e)
    finally:
        state["running"], state["progress"], state["stage"] = False, 1.0, "Gotowe"


@app.post("/api/scan")
def start(req: ScanReq):
    with lock:
        if state["running"]:
            return {"started": False, **state}
        state.update(running=True, progress=0.0, stage="Start", error=None, started=time.time())
        threading.Thread(target=worker, args=(req,), daemon=True).start()
    return {"started": True, **state}


@app.get("/api/status")
def status():
    return {**state, "generated_at": scan["generated_at"] if scan else None}


def lite(r):
    return {k: v for k, v in r.items() if k not in ("chart", "fvg_list")}


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
                return r
    raise HTTPException(404, "Brak spółki w ostatnim skanie")


@app.get("/api/export.csv")
def export():
    df = scanner.to_rows(scan) if scan else None
    buf = io.StringIO()
    if df is not None:
        df.to_csv(buf, index=False, sep=";", decimal=",")
    return StreamingResponse(iter(["\ufeff" + buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=strat_scan.csv"})


# --- interfejs (samodzielna aplikacja) ---
WEB = os.path.join(HERE, "web")


@app.get("/")
def index():
    return FileResponse(os.path.join(WEB, "index.html"))


if os.path.isdir(WEB):
    app.mount("/static", StaticFiles(directory=WEB), name="static")
