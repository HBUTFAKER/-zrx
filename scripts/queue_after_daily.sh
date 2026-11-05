#!/bin/bash
# 日线 → 财务 → 清洗 的串行接续（不是探测脚本，纯粹按队列执行）
# 判定：日线日志出现 [done] 才接财务；出现 [STOP]（封禁）则终止并留记录。
set -u
cd "C:/Users/29212/quant-app" || exit 1
PY=".venv/Scripts/python.exe"
DLOG="data/raw/_state/daily_run2.log"
FLOG="data/raw/_state/fin_run2.log"
QLOG="data/raw/_state/queue.log"

echo "[$(date '+%m-%d %H:%M')] queue start, 等日线" >> "$QLOG"

# 最多等 5 小时
for i in $(seq 1 300); do
  if grep -q '\[done\]' "$DLOG" 2>/dev/null; then
    echo "[$(date '+%m-%d %H:%M')] 日线 done" >> "$QLOG"
    break
  fi
  if grep -q '\[STOP\]' "$DLOG" 2>/dev/null; then
    echo "[$(date '+%m-%d %H:%M')] 日线被封/中止，不接财务" >> "$QLOG"
    exit 2
  fi
  sleep 60
done

if ! grep -q '\[done\]' "$DLOG" 2>/dev/null; then
  echo "[$(date '+%m-%d %H:%M')] 等待超时或异常退出" >> "$QLOG"
  exit 3
fi

echo "[$(date '+%m-%d %H:%M')] 开始财务 2018-2022 workers=1" >> "$QLOG"
"$PY" -u -m src.ingest --stage finance --years 2018 2022 --workers 1 --chunk 60 >> "$FLOG" 2>&1
frc=$?
echo "[$(date '+%m-%d %H:%M')] 财务 rc=$frc" >> "$QLOG"

if [ $frc -eq 0 ]; then
  echo "[$(date '+%m-%d %H:%M')] 开始清洗" >> "$QLOG"
  "$PY" -u -m src.clean --years 2018 2024 >> data/raw/_state/clean_run2.log 2>&1
  echo "[$(date '+%m-%d %H:%M')] 清洗 rc=$?" >> "$QLOG"
fi
echo "[$(date '+%m-%d %H:%M')] queue exit" >> "$QLOG"
