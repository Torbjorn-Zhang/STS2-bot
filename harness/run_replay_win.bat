@echo off
chcp 65001 >nul
set PYTHONUTF8=1
D:\sts2\.venv\Scripts\python.exe "%~dp0run_replay_win.py" %*
pause
