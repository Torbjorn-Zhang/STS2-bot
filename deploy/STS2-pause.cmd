@echo off
rem Take this machine back from STS2 training: stop every training process now and keep the
rem fleet coordinator from sending more work until STS2-resume.cmd is run.
echo paused> D:\sts2\PAUSE
powershell -NoProfile -Command "$r='D:\sts2'; Get-CimInstance Win32_Process -Filter \"Name='cmd.exe'\" | Where-Object { $_.CommandLine -like \"*$r\bot\fleet\jobs\*\" } | ForEach-Object { taskkill /PID $_.ProcessId /T /F 2>&1 | Out-Null }; Get-CimInstance Win32_Process | Where-Object { $_.ExecutablePath -and $_.ExecutablePath.StartsWith($r, 'OrdinalIgnoreCase') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
echo STS2 training stopped on this machine. Run STS2-resume.cmd to lend it again.
pause
