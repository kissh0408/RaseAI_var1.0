@echo off
REM Weekly 0B41/0B42 time-series odds auto-fetch task (Windows Task Scheduler).
REM Retention is 1 year, so a missed run means permanent data loss for that week.
REM Assumes main/data/race/race_ra.csv is kept current by main/unified_pipeline.py
REM (same assumption as the existing 0B31 fetcher in this repo).

setlocal enabledelayedexpansion
cd /d "C:\Users\syugo\AI\RaceAI_var1.0"

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd"') do set END_DATE=%%i
for /f %%i in ('powershell -NoProfile -Command "(Get-Date).AddDays(-7) | Get-Date -Format yyyyMMdd"') do set START_DATE=%%i

set LOG_DIR=common\data\output\odds_ts\logs
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
set LOG_FILE=%LOG_DIR%\weekly_%END_DATE%.log

echo [%date% %time%] weekly odds_ts fetch: %START_DATE% - %END_DATE% > "%LOG_FILE%"
py -3-32 common\data\src\fetch_odds_ts_cli.py weekly --start-date %START_DATE% --end-date %END_DATE% >> "%LOG_FILE%" 2>&1
set EXIT_CODE=%ERRORLEVEL%
echo [%date% %time%] exit code: %EXIT_CODE% >> "%LOG_FILE%"

exit /b %EXIT_CODE%
