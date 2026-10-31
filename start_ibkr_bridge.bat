@echo off
title Strat Scanner - most IBKR
cd /d "%~dp0"
python -c "import ib_insync" 2>nul || (
  echo Instaluje ib_insync...
  python -m pip install ib_insync
)
python ibkr_bridge.py
pause
