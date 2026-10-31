"""Pamięć podręczna notowań dziennych (Yahoo Finance).

Pierwsze pobranie: 10 lat historii dla wszystkich spółek.
Kolejne odświeżenia: tylko ostatnie ~10 dni, doklejane do historii — dzięki temu skan trwa sekundy, a nie minuty.
"""
from __future__ import annotations

import os
import pickle
import time
from datetime import datetime
from zoneinfo import ZoneInfo
WAW = ZoneInfo("Europe/Warsaw")

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "data_cache.pkl")

STORE: dict = {"data": {}, "full_at": None, "updated_at": None}


def _load_disk():
    if STORE["data"] or not os.path.exists(PATH):
        return
    try:
        with open(PATH, "rb") as f:
            STORE.update(pickle.load(f))
    except Exception:
        pass


def _save_disk():
    try:
        with open(PATH + ".tmp", "wb") as f:
            pickle.dump(STORE, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(PATH + ".tmp", PATH)
    except Exception:
        pass


def _download(tickers, period, progress_cb=None, chunk=80):
    import yfinance as yf
    out = {}
    chunks = [tickers[i:i + chunk] for i in range(0, len(tickers), chunk)]
    for ci, ch in enumerate(chunks):
        df = None
        for attempt in range(3):
            try:
                df = yf.download(ch, period=period, interval="1d", group_by="ticker",
                                 auto_adjust=False, progress=False, threads=True)
                break
            except Exception:
                time.sleep(2 + attempt * 2)
        if df is not None and not df.empty:
            for t in ch:
                try:
                    sub = df[t] if isinstance(df.columns, pd.MultiIndex) else df
                    sub = sub[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Open", "High", "Low", "Close"])
                    if len(sub):
                        sub.index = pd.to_datetime(sub.index).tz_localize(None)
                        out[t] = sub.astype("float64")
                except Exception:
                    pass
        if progress_cb:
            progress_cb((ci + 1) / len(chunks))
    return out


def get_data(tickers: list[str], progress_cb=None, force_full: bool = False) -> dict:
    """Zwraca {ticker: DataFrame dzienny}. Dociąga tylko brakujące dni, jeśli historia jest już w pamięci."""
    _load_disk()
    data = STORE["data"]
    missing = [t for t in tickers if t not in data]
    have = [t for t in tickers if t in data]
    if force_full:
        missing, have = list(tickers), []
    n_steps = (1 if missing else 0) + (1 if have else 0)
    step = [0]

    def sub_cb(p):
        if progress_cb:
            progress_cb((step[0] + p) / max(1, n_steps))

    if missing:
        full = _download(missing, "10y", sub_cb, chunk=60)
        for t, df in full.items():
            if len(df) > 60:
                data[t] = df
        if len(missing) > 50:
            STORE["full_at"] = datetime.now(WAW).strftime("%Y-%m-%d %H:%M")
        step[0] += 1
    if have:
        inc = _download(have, "10d", sub_cb, chunk=120)
        refetch = []
        for t, new in inc.items():
            old = data.get(t)
            if old is None or new.empty:
                continue
            # split / korekta historii: zamknięcie z tego samego dnia różni się > 2% -> pełne pobranie
            common = old.index.intersection(new.index)
            if len(common):
                d0 = common[0]
                if abs(old.at[d0, "Close"] / new.at[d0, "Close"] - 1) > 0.02:
                    refetch.append(t)
                    continue
            data[t] = pd.concat([old[old.index < new.index.min()], new])
        if refetch:
            for t, df in _download(refetch, "10y", None, chunk=60).items():
                if len(df) > 60:
                    data[t] = df
        step[0] += 1
    STORE["updated_at"] = datetime.now(WAW).strftime("%Y-%m-%d %H:%M")
    _save_disk()
    return {t: data[t] for t in tickers if t in data}


def info() -> dict:
    return {"tickers": len(STORE["data"]), "full_at": STORE["full_at"], "updated_at": STORE["updated_at"]}
