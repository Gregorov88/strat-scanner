---
title: Strat Scanner
emoji: 📈
colorFrom: gray
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# Strat Scanner — Nasdaq 100 + S&P 500

Skaner setupów The Strat (2D-2U, 2U-2U, 2U-1-2U, F2D / 2U-2D, 2D-2D, 2D-1-2D, F2U)
z oceną A/B/C wg FTFC, EMA10/20 i FVG. Dane: Yahoo Finance.

Uruchomienie lokalnie:

    pip install -r requirements.txt
    uvicorn server:app --port 7860

i otwórz http://localhost:7860
