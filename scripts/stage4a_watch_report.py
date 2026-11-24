"""Refresh the evidence report as a long training run progresses and ends."""
import json,time
from pathlib import Path
from stage4a_report import RUN,main
last=-1
while True:
    csv_path=RUN/'results.csv'
    if csv_path.exists():
        rows=max(0,len(csv_path.read_text(encoding='utf-8-sig').splitlines())-1)
        if rows!=last:
            try:main()
            except Exception:pass
            last=rows
    p=RUN/'stage4a_provenance.json'
    if p.exists() and json.loads(p.read_text()).get('outcome') in ('completed','interrupted_or_failed'):
        main();break
    time.sleep(30)
