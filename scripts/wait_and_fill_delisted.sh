#!/bin/bash
# 等 baostock 解封 → 补 226 只退市股财务 → 重跑清洗 → 跑测试
# 这次等待上限放到 24 小时（上一版设 5 小时，日线跑了 5h12m 直接超时废掉）
# 探测间隔 20 分钟：探测太勤没意义，还可能在封禁窗口内反复刷新计数。
set -u
cd "C:/Users/29212/quant-app" || exit 1
PY=".venv/Scripts/python.exe"
LOG="data/raw/_state/wait_delisted.log"

probe() {
  timeout 90 "$PY" -c "
import baostock as bs
lg = bs.login()
ok = (lg.error_code == '0')
if ok:
    bs.logout()
print('OK' if ok else 'BAN')
" 2>/dev/null | grep -oE 'OK|BAN' | head -1
}

echo "[$(date '+%m-%d %H:%M')] watcher start" >> "$LOG"
for i in $(seq 1 72); do
  st=$(probe)
  echo "[$(date '+%m-%d %H:%M')] try $i -> ${st:-UNKNOWN}" >> "$LOG"
  if [ "$st" = "OK" ]; then
    echo "[$(date '+%m-%d %H:%M')] 解封，开始补退市股（限速 1 秒/次）" >> "$LOG"
    "$PY" -u -m src.ingest --stage finance --delisted-only --req-interval 1 \
        --years 2018 2022 --workers 1 --chunk 30 >> data/raw/_state/delisted_fin.log 2>&1
    rc=$?
    echo "[$(date '+%m-%d %H:%M')] 补抓 rc=$rc" >> "$LOG"
    if [ $rc -eq 0 ]; then
      "$PY" -u -m src.clean --years 2018 2024 >> data/raw/_state/clean_run3.log 2>&1
      echo "[$(date '+%m-%d %H:%M')] 清洗 rc=$?" >> "$LOG"
      "$PY" -m pytest -q > data/raw/_state/pytest_final.log 2>&1
      echo "[$(date '+%m-%d %H:%M')] pytest rc=$?" >> "$LOG"
    else
      echo "[$(date '+%m-%d %H:%M')] 补抓失败，可能是又被封，等人工处理" >> "$LOG"
    fi
    break
  fi
  sleep 1200
done
echo "[$(date '+%m-%d %H:%M')] watcher exit" >> "$LOG"
