"""Backtest FVG + Strat (zadania 1.1 i 2.1).

1.1  Wejście w trendzie przy powrocie do niewypełnionego FVG (D / W / M).
     Trend: EMA10 > EMA20 (long) / EMA10 < EMA20 (short) na interwale FVG, liczone na poprzedniej zamkniętej świecy.
     Wejście: pierwszy dotyk krawędzi FVG (long: górna krawędź byczego FVG), symulowane na świecach dziennych.
     Cel: ekstremum ruchu od powstania FVG do powrotu (szczyt przed korektą / dołek przed odbiciem).
     Negacja: zamknięcie świecy interwału FVG poza dalszą krawędzią FVG (jak „Close” we wskaźniku FVG/iFVG).
     Wynik: Zysk (cel przed negacją) / Negacja / Otwarte (ani jedno w horyzoncie).
     MFE: maksymalny ruch w $ i % od wejścia do negacji lub końca horyzontu („ile dało się zebrać”).

2.1  Jak 1.1 (FVG W lub M), ale wejście dopiero po setupie na niższym interwale (W -> D, M -> W):
     świeca sygnałowa 2D lub F2D (long) / 2U lub F2U (short), która dotknęła strefy FVG,
     trigger = przebicie jej high (low) przez następną świecę, SL = low (high) świecy sygnałowej.
     Liczymy, czy cena doszła do 1R i 2R przed SL (konserwatywnie: SL i cel w tym samym dniu = strata),
     oraz czy doszła do najbliższego swing pivota.
"""
from __future__ import annotations

import time
from collections import defaultdict

import numpy as np
import pandas as pd

from scanner import resample, candle_label, ema, find_pivot_tp

VERSION = 2
HORIZON = {"D": 60, "W": 52, "M": 36}          # maks. liczba świec interwału FVG na realizację
MAX_AGE = {"D": 120, "W": 150, "M": 120}       # maks. wiek FVG (świece) w chwili powrotu
CONFIRM = {"W": "D", "M": "W"}                 # interwał setupu Strat dla 2.1
CONFIRM_WIN = {"W": 15, "M": 13}               # ile świec niższego TF po dotyku szukamy setupu
STRAT_HORIZON_DAYS = {"W": 60, "M": 130}       # horyzont trade'u 2.1 w dniach sesyjnych


def _period_bounds(xidx: pd.DatetimeIndex, didx: pd.DatetimeIndex, tf: str):
    """Dla każdej świecy interwału tf: [indeks pierwszego dnia, indeks ostatniego dnia] w danych dziennych."""
    if tf == "D":
        a = np.arange(len(didx))
        return a, a
    ends = np.searchsorted(didx.values, xidx.values, side="right") - 1
    starts = np.r_[0, ends[:-1] + 1]
    return starts, ends


def _fvg_events(t, d, tf, m_trend_at):
    """Zdarzenia 1.1 dla jednej spółki i jednego interwału."""
    xb = resample(d, tf)
    if len(xb) < 30:
        return []
    o, h, l, c = (xb[k].values for k in ("Open", "High", "Low", "Close"))
    e10, e20 = ema(xb["Close"], 10).values, ema(xb["Close"], 20).values
    didx = d.index
    do, dh, dl, dc = (d[k].values for k in ("Open", "High", "Low", "Close"))
    st, en = _period_bounds(xb.index, didx, tf)
    n = len(xb)
    ev = []
    for i in range(2, n - 1):
        for kind in ("bull", "bear"):
            if kind == "bull":
                if not h[i - 2] < l[i]:
                    continue
                top, bot = l[i], h[i - 2]
            else:
                if not l[i - 2] > h[i]:
                    continue
                top, bot = l[i - 2], h[i]
            # pierwszy powrót do strefy
            j = None
            for q in range(i + 1, min(n, i + 1 + MAX_AGE[tf])):
                if (kind == "bull" and l[q] <= top) or (kind == "bear" and h[q] >= bot):
                    j = q
                    break
            if j is None or j < 21:
                continue
            up = e10[j - 1] > e20[j - 1]
            if (kind == "bull") != up:
                continue                      # tylko w trendzie
            sign = 1 if kind == "bull" else -1
            edge, far = (top, bot) if kind == "bull" else (bot, top)
            target = float(np.max(h[i:j])) if kind == "bull" else float(np.min(l[i:j]))
            if sign * (target - edge) <= 0:
                continue
            # dzień wejścia w obrębie świecy j
            t0 = None
            for q in range(st[j], en[j] + 1):
                if (kind == "bull" and dl[q] <= edge) or (kind == "bear" and dh[q] >= edge):
                    t0 = q
                    break
            if t0 is None:
                continue
            entry = float(min(edge, do[t0])) if kind == "bull" else float(max(edge, do[t0]))
            # negacja: zamknięcie świecy tf poza dalszą krawędzią
            neg_k = None
            for k in range(j, min(n, j + HORIZON[tf] + 1)):
                if (kind == "bull" and c[k] < far) or (kind == "bear" and c[k] > far):
                    neg_k = k
                    break
            last_k = neg_k if neg_k is not None else min(n - 1, j + HORIZON[tf])
            end_day = en[last_k]
            win_day = None
            mfe = 0.0
            for q in range(t0 + 1, end_day + 1):
                fav = (dh[q] - entry) if kind == "bull" else (entry - dl[q])
                if fav > mfe:
                    mfe = fav
                if win_day is None and ((kind == "bull" and dh[q] >= target) or (kind == "bear" and dl[q] <= target)):
                    win_day = q
            # świeca dotyku sama może zakończyć się poniżej strefy (neg_k == j) — wtedy liczy się tylko to, co po wejściu
            if win_day is not None:
                outcome = "win"
            elif neg_k is not None:
                outcome = "neg"
            else:
                outcome = "open" if last_k >= n - 1 else "neg_time"   # nie doszło do celu w horyzoncie
            risk = abs(entry - far)
            date = didx[t0]
            ev.append({
                "t": t, "tf": tf, "dir": "long" if kind == "bull" else "short", "date": str(date.date()),
                "y": int(date.year), "zone": [round(float(bot), 2), round(float(top), 2)],
                "entry": round(entry, 2), "target": round(target, 2), "outcome": outcome,
                "days": int(win_day - t0) if win_day is not None else None,
                "mfe": round(mfe, 2), "mfe_pct": round(mfe / entry * 100, 2),
                "mfe_r": round(mfe / risk, 2) if risk > 0 else None,
                "tgt_pct": round(abs(target - entry) / entry * 100, 2),
                "m_al": m_trend_at(date, kind) if tf != "M" else None,
                "win_date": str(didx[win_day].date()) if win_day is not None else None,
                "_j": j, "_t0": t0, "_edge": float(edge), "_far": float(far), "_top": float(top), "_bot": float(bot),
                "_neg_day": int(en[neg_k]) if neg_k is not None else None,
            })
    return ev


def _strat_events(t, d, tf, fe, m_trend_at):
    """Zdarzenia 2.1 dla FVG z interwału tf (W lub M) i setupu Strat na interwale niższym."""
    ytf = CONFIRM[tf]
    yb = resample(d, ytf)
    yo, yh, yl, yc = (yb[k].values for k in ("Open", "High", "Low", "Close"))
    didx = d.index
    dh, dl, do = d["High"].values, d["Low"].values, d["Open"].values
    yst, yen = _period_bounds(yb.index, didx, ytf)
    ny = len(yb)
    # indeks świecy Y zawierającej dany dzień
    day2y = np.searchsorted(yen, np.arange(len(didx)), side="left")
    out = []
    for e in fe:
        if e["tf"] != tf:
            continue
        kind = "bull" if e["dir"] == "long" else "bear"
        k0 = int(day2y[e["_t0"]])
        stop_day = e["_neg_day"] if e["_neg_day"] is not None else len(didx) - 1
        found = None
        for k in range(max(1, k0), min(ny - 1, k0 + CONFIRM_WIN[tf])):
            if yst[k] > stop_day:
                break
            lab = candle_label(yo[k], yh[k], yl[k], yc[k], yh[k - 1], yl[k - 1])
            if kind == "bull":
                if lab not in ("2D", "F2D") or yl[k] > e["_top"] or yc[k] < e["_bot"]:
                    continue
                if yh[k + 1] > yh[k]:
                    found = (k, lab)
                    break
            else:
                if lab not in ("2U", "F2U") or yh[k] < e["_bot"] or yc[k] > e["_top"]:
                    continue
                if yl[k + 1] < yl[k]:
                    found = (k, lab)
                    break
        if not found:
            continue
        k, lab = found
        pattern = ("2D-2U" if lab == "2D" else "F2D") if kind == "bull" else ("2U-2D" if lab == "2U" else "F2U")
        trig = float(yh[k]) if kind == "bull" else float(yl[k])
        sl = float(yl[k]) if kind == "bull" else float(yh[k])
        # dzień wejścia w świecy k+1
        t0 = None
        for q in range(yst[k + 1], yen[k + 1] + 1):
            if (kind == "bull" and dh[q] > trig) or (kind == "bear" and dl[q] < trig):
                t0 = q
                break
        if t0 is None:
            continue
        entry = max(trig, float(do[t0])) if kind == "bull" else min(trig, float(do[t0]))
        risk = abs(entry - sl)
        if risk <= 0:
            continue
        piv, _ = find_pivot_tp(yh[:k + 1], yl[:k + 1], k, entry, "long" if kind == "bull" else "short")
        sign = 1 if kind == "bull" else -1
        r1, r2 = entry + sign * risk, entry + sign * 2 * risk
        hit = {"r1": None, "r2": None, "piv": None}
        sl_day = None
        max_r = 0.0
        for q in range(t0, min(len(didx), t0 + STRAT_HORIZON_DAYS[tf])):
            hi, lo = dh[q], dl[q]
            fav = (hi - entry) if kind == "bull" else (entry - lo)
            stop = (lo <= sl) if kind == "bull" else (hi >= sl)
            if stop:
                sl_day = q
                break                       # konserwatywnie: SL i cel tego samego dnia = najpierw SL
            max_r = max(max_r, fav / risk)
            for key, lvl in (("r1", r1), ("r2", r2), ("piv", piv)):
                if lvl is not None and hit[key] is None and sign * ((hi if kind == "bull" else lo) - lvl) >= 0:
                    hit[key] = q
        date = didx[t0]
        out.append({
            "t": t, "tf": tf, "ytf": ytf, "dir": e["dir"], "pattern": pattern, "date": str(date.date()),
            "y": int(date.year), "entry": round(entry, 2), "sl": round(sl, 2), "risk_pct": round(risk / entry * 100, 2),
            "rr1": hit["r1"] is not None, "rr2": hit["r2"] is not None,
            "piv": round(float(piv), 2) if piv is not None else None, "piv_hit": hit["piv"] is not None,
            "piv_rr": round(abs(piv - entry) / risk, 2) if piv is not None else None,
            "sl_hit": sl_day is not None, "max_r": round(max_r, 2),
            "days_r1": int(hit["r1"] - t0) if hit["r1"] is not None else None,
            "m_al": e["m_al"], "fvg_outcome": e["outcome"],
            # transakcja jeszcze trwa (brak SL i 2R, a dane się skończyły) — nie wliczamy do statystyk
            "open": bool(sl_day is None and hit["r2"] is None and t0 + STRAT_HORIZON_DAYS[tf] > len(didx)),
        })
    return out


def _m_trend_fn(d):
    mb = resample(d, "M")
    up = (ema(mb["Close"], 10) > ema(mb["Close"], 20)).values
    idx = mb.index.values

    def f(date, kind):
        k = np.searchsorted(idx, np.datetime64(date), side="left") - 1   # ostatni zamknięty miesiąc
        if k < 20:
            return None
        return bool(up[k]) == (kind == "bull")
    return f


FVG_COLS = ["t", "tf", "dir", "date", "y", "entry", "target", "outcome", "days", "mfe", "mfe_pct", "mfe_r",
            "tgt_pct", "m_al", "after_m"]
STRAT_COLS = ["t", "tf", "ytf", "dir", "pattern", "date", "y", "entry", "sl", "risk_pct", "rr1", "rr2", "piv",
              "piv_hit", "piv_rr", "sl_hit", "max_r", "days_r1", "m_al", "fvg_outcome", "open"]


CODES = {"tf": ["D", "W", "M"], "ytf": ["D", "W", "M"], "dir": ["long", "short"],
         "outcome": ["win", "neg", "neg_time", "open"], "fvg_outcome": ["win", "neg", "neg_time", "open"],
         "pattern": ["2D-2U", "F2D", "2U-2D", "F2U"]}
F32 = ("entry", "target", "mfe", "mfe_pct", "mfe_r", "tgt_pct", "sl", "risk_pct", "piv", "piv_rr", "max_r",
       "days", "days_r1", "m_al", "after_m")


def _compact(rows, cols, tid):
    """Lista słowników -> mały DataFrame liczbowy (kody zamiast tekstu), żeby 300 tys. zdarzeń mieściło się w RAM."""
    df = pd.DataFrame(rows, columns=cols)
    df["t"] = np.int16(tid)
    for c, vals in CODES.items():
        if c in df.columns:
            df[c] = df[c].map({v: i for i, v in enumerate(vals)}).fillna(-1).astype("int8")
    for c in F32:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].map(lambda v: np.nan if v is None else float(v)), errors="coerce").astype("float32")
    for c in ("rr1", "rr2", "piv_hit", "sl_hit", "open"):
        if c in df.columns:
            df[c] = df[c].astype(bool)
    df["date"] = pd.to_datetime(df["date"])
    df["y"] = df["y"].astype("int16")
    return df


def _expand(df, names):
    """Kody -> Categorical (bez kopiowania tekstu)."""
    if df.empty:
        return df
    df["t"] = pd.Categorical.from_codes(df["t"].astype("int32"), categories=names)
    for c, vals in CODES.items():
        if c in df.columns:
            df[c] = pd.Categorical.from_codes(df[c].astype("int8"), categories=vals)
    return df


def run(data: dict, progress_cb=None) -> dict:
    t0 = time.time()
    fparts, sparts, names = [], [], []
    items = list(data.items())
    for n, (t, d) in enumerate(items):
        try:
            if len(d) < 300:
                continue
            mt = _m_trend_fn(d)
            fe = []
            for tf in ("D", "W", "M"):
                fe.extend(_fvg_events(t, d, tf, mt))
            se = []
            for tf in ("W", "M"):
                se.extend(_strat_events(t, d, tf, fe, mt))
            # zależność między interwałami: W po zrealizowanym M (ten sam kierunek, do 6 mies. po celu M)
            m_wins = [(e["dir"], pd.Timestamp(e["win_date"])) for e in fe if e["tf"] == "M" and e["outcome"] == "win"]
            for e in fe:
                e["after_m"] = None
                if e["tf"] == "W":
                    dt = pd.Timestamp(e["date"])
                    e["after_m"] = any(dr == e["dir"] and wd < dt <= wd + pd.Timedelta(days=183) for dr, wd in m_wins)
            tid = len(names)
            names.append(t)
            if fe:
                fparts.append(_compact([{c: e[c] for c in FVG_COLS} for e in fe], FVG_COLS, tid))
            if se:
                sparts.append(_compact(se, STRAT_COLS, tid))
        except Exception:
            pass
        if progress_cb and n % 20 == 0:
            progress_cb((n + 1) / len(items))
    fvg = pd.concat(fparts, ignore_index=True) if fparts else pd.DataFrame(columns=FVG_COLS)
    strat = pd.concat(sparts, ignore_index=True) if sparts else pd.DataFrame(columns=STRAT_COLS)
    return {"v": VERSION, "fvg": _expand(fvg, names), "strat": _expand(strat, names), "elapsed_s": round(time.time() - t0, 1),
            "generated_at": pd.Timestamp.now(tz="Europe/Warsaw").strftime("%Y-%m-%d %H:%M")}


# ----------------------------------------------------------------------------------------------
# Agregacja (na DataFrame'ach — mało pamięci przy ~300 tys. zdarzeń)
# ----------------------------------------------------------------------------------------------

def _pct(a, b):
    return round(float(a) / b * 100, 1) if b else None


def _med(s):
    s = pd.Series(s).dropna()
    return round(float(s.median()), 2) if len(s) else None


def _avg(s):
    s = pd.Series(s).dropna()
    return round(float(s.mean()), 2) if len(s) else None


def _fvg_stats(ev: pd.DataFrame):
    n = len(ev)
    oc = ev["outcome"].astype(str)
    closed = int((oc != "open").sum())
    w = int((oc == "win").sum())
    ng = int((oc == "neg").sum())
    yrs = ev["y"].nunique()
    return {"n": n, "win": w, "neg": ng, "win_pct": _pct(w, closed), "neg_pct": _pct(ng, closed),
            "open": n - closed, "nt_pct": _pct(int((oc == "neg_time").sum()), closed),
            "mfe_pct_med": _med(ev["mfe_pct"]), "mfe_pct_avg": _avg(ev["mfe_pct"]),
            "mfe_usd_avg": _avg(ev["mfe"]), "mfe_r_med": _med(ev["mfe_r"]),
            "tgt_pct_med": _med(ev["tgt_pct"]), "days_med": _med(ev.loc[oc == "win", "days"]),
            "per_year": round(n / max(1, yrs), 1)}


def _strat_stats(ev: pd.DataFrame):
    n_open = int(ev["open"].sum()) if len(ev) else 0
    ev = ev[~ev["open"]] if len(ev) else ev
    n = len(ev)
    p1 = int(ev["rr1"].sum()) if n else 0
    p2 = int(ev["rr2"].sum()) if n else 0
    pv = ev[ev["piv"].notna()]
    return {"n": n, "rr1": p1, "rr1_pct": _pct(p1, n), "rr2_pct": _pct(p2, n),
            "exp1": round((2 * p1 - n) / n, 2) if n else None,          # oczekiwana wartość w R przy TP 1R
            "exp2": round((3 * p2 - n) / n, 2) if n else None,          # przy TP 2R
            "piv_pct": _pct(int(pv["piv_hit"].sum()), len(pv)), "piv_rr_med": _med(pv["piv_rr"]),
            "max_r_med": _med(ev["max_r"]), "days_r1_med": _med(ev["days_r1"]),
            "risk_pct_med": _med(ev["risk_pct"]), "open": n_open}


def _records(df):
    out = df.astype(object).where(df.notna(), None).to_dict("records")
    for r in out:
        for k, v in r.items():
            if isinstance(v, pd.Timestamp):
                r[k] = str(v.date())
            elif isinstance(v, float):
                r[k] = round(v, 2)
            elif isinstance(v, (np.bool_,)):
                r[k] = bool(v)
            elif isinstance(v, (np.integer,)):
                r[k] = int(v)
    return out


def aggregate(bt: dict, tickers: set | None = None, years: int | None = None) -> dict:
    fe, se = bt["fvg"], bt["strat"]
    if tickers:
        fe = fe[fe["t"].astype(str).isin(tickers)]
        se = se[se["t"].astype(str).isin(tickers)]
    if years:
        y0 = pd.Timestamp.now().year - years + 1
        fe = fe[fe["y"] >= y0]
        se = se[se["y"] >= y0]
    ftf, fdir = fe["tf"].astype(str), fe["dir"].astype(str)
    t11 = [{"tf": tf, "dir": dr, **_fvg_stats(fe[(ftf == tf) & (fdir == dr)])} for tf in ("M", "W", "D") for dr in ("long", "short")]
    mtf = []
    for dr in ("long", "short"):
        w_all = fe[(ftf == "W") & (fdir == dr)]
        mtf.append({"dir": dr, "label": "W po zrealizowanym M", **_fvg_stats(w_all[w_all["after_m"] == True])})  # noqa: E712
        mtf.append({"dir": dr, "label": "W zgodne z trendem M", **_fvg_stats(w_all[w_all["m_al"] == True])})  # noqa: E712
        mtf.append({"dir": dr, "label": "W przeciw trendowi M", **_fvg_stats(w_all[w_all["m_al"] == False])})  # noqa: E712
    stf, sdir, spat = se["tf"].astype(str), se["dir"].astype(str), se["pattern"].astype(str)
    t21 = []
    for tf in ("M", "W"):
        for dr in ("long", "short"):
            m = (stf == tf) & (sdir == dr)
            ev = se[m]
            row = {"tf": tf, "ytf": CONFIRM[tf], "dir": dr, **_strat_stats(ev)}
            row["by_pattern"] = {p: _strat_stats(ev[spat[m] == p]) for p in sorted(spat[m].unique())}
            if tf == "W":
                row["m_al"] = _strat_stats(ev[ev["m_al"] == True])  # noqa: E712
                row["m_against"] = _strat_stats(ev[ev["m_al"] == False])  # noqa: E712
            t21.append(row)
    by_year = defaultdict(lambda: {"D": 0, "W": 0, "M": 0, "W→D": 0, "M→W": 0})
    for (y, tf), c in fe.groupby([fe["y"], ftf]).size().items():
        by_year[int(y)][tf] += int(c)
    for (y, tf), c in se.groupby([se["y"], stf]).size().items():
        by_year[int(y)][f"{tf}→{CONFIRM[tf]}"] += int(c)
    fw = fe[ftf != "D"]
    oc = fw["outcome"].astype(str)
    g11 = pd.DataFrame({"t": fw["t"].astype(str), "n": 1, "c": (oc != "open").astype(int), "w": (oc == "win").astype(int)}).groupby("t").sum()
    g21 = pd.DataFrame({"t": se["t"].astype(str), "n": 1, "r": se["rr1"].astype(int)}).groupby("t").sum()
    allt = sorted(set(g11.index) | set(g21.index))
    tick = []
    for t in allt:
        a = g11.loc[t] if t in g11.index else None
        b = g21.loc[t] if t in g21.index else None
        tick.append({"t": t, "n11": int(a["n"]) if a is not None else 0,
                     "win11": _pct(a["w"], a["c"]) if a is not None else None,
                     "n21": int(b["n"]) if b is not None else 0, "rr1": _pct(b["r"], b["n"]) if b is not None else None})
    tick.sort(key=lambda x: (-x["n21"], x["t"]))
    recent = _records(se.sort_values("date", ascending=False).head(60))
    recent11 = _records(fw.sort_values("date", ascending=False).head(60))
    return {"generated_at": bt.get("generated_at"), "n_tickers": int(fe["t"].astype(str).nunique()),
            "t11": t11, "mtf": mtf, "t21": t21, "by_year": [{"y": y, **v} for y, v in sorted(by_year.items())],
            "tickers": tick[:600], "recent21": recent, "recent11": recent11}
