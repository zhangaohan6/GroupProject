#!/bin/bash
cd ~/Desktop/COMP9517-GroupProject
until grep -q "ALL DONE" logs/fetch.log 2>/dev/null; do sleep 60; done
N=$(find data/images_256 -type f 2>/dev/null | wc -l | tr -d ' ')
SZ=$(du -sh data/images_256 2>/dev/null | awk '{print $1}')
python3 - "$N" "$SZ" <<'PY'
import sys
p="PLAN.md"; t=open(p).read().rstrip()+"\n"
t+=f"- 2026-07-20 M1 图片抽取完成:images_256 共 {sys.argv[1]} 张({sys.argv[2]}),256px。下一步 A 上传 data/ 到共享 Drive → 组员开训。\n"
open(p,"w").write(t)
PY
git -c user.name="zhangaohan6" -c user.email="youyuxs@gmail.com" commit -q -am "M1 complete: fetched ${N} images (256px); progress logged"
git push origin main >> logs/fetch.log 2>&1
echo "[finalize $(date '+%H:%M:%S')] DONE + pushed. images=${N} size=${SZ}" >> logs/fetch.log
