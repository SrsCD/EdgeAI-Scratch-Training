"""Offline, localhost-only detection history and CSV export."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO
from pathlib import Path
import sqlite3
from urllib.parse import parse_qs, urlencode, urlsplit
from uuid import UUID

from stage5h_events import DEFAULT_DB, NAMES, read_events


def csv_text(rows):
    fields = ("event_id", "timestamp_utc", "class_id", "class_name", "confidence",
              "x1", "y1", "x2", "y2", "source_id", "checkpoint_id", "event_type",
              "evidence_path", "notification_status", "sync_status")
    out = StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def html_page(all_rows, rows, selected, start, end):
    counts = Counter(row["class_name"] for row in all_rows)
    options = '<option value="">All classes</option>' + "".join(
        f'<option value="{name}" {"selected" if name == selected else ""}>{name}</option>' for name in NAMES)
    query = urlencode({"class": selected, "start": start, "end": end})
    body = []
    for row in rows:
        event_id = escape(str(row["event_id"]))
        preview = f'<a href="/evidence/{event_id}" target="_blank"><img src="/evidence/{event_id}" alt="Evidence" width="100"></a>' if row.get("evidence_path") else "—"
        cells = (row["timestamp_utc"], row["class_name"], f'{row["confidence"]:.3f}',
                 row["source_id"], row["event_type"], row["notification_status"], row["sync_status"])
        body.append("<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in cells) + f"<td>{preview}</td></tr>")
    summary = " ".join(f"<span>{name}: {counts[name]}</span>" for name in NAMES)
    return f'''<!doctype html><html><head><meta charset="utf-8"><title>Model B detection history</title>
<style>body{{font:15px system-ui;margin:2rem;background:#101720;color:#eef3f8}}
input,select,button,a{{font:inherit}} input,select,button{{padding:.4rem;margin:.2rem}}
a{{color:#91d9ff}} .cards span{{display:inline-block;background:#263a4a;padding:.7rem;margin:.3rem;border-radius:5px}}
table{{border-collapse:collapse;width:100%;margin-top:1rem}}td,th{{border-bottom:1px solid #405364;padding:.5rem;text-align:left}}
small{{color:#b6c5d0}}</style></head><body><h1>Model B — Detection history</h1>
<p><strong>Total events: {len(all_rows)}</strong> · Filtered events: {len(rows)}</p>
<div class="cards">{summary}</div><p><small>Model detections are unverified observations, not verified security threats. Local-only history.</small></p>
<form method="get" action="/"><label>Class <select name="class">{options}</select></label>
<label>From (UTC) <input type="date" name="start" value="{escape(start)}"></label>
<label>Through (UTC) <input type="date" name="end" value="{escape(end)}"></label>
<button type="submit">Filter</button> <a href="/export.csv?{query}">Export filtered CSV</a></form>
<table><thead><tr><th>UTC timestamp</th><th>Class</th><th>Confidence</th><th>Source</th><th>Event type</th><th>Notification</th><th>Sync</th><th>Evidence</th></tr></thead>
<tbody>{''.join(body)}</tbody></table></body></html>'''


def make_handler(database: Path):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            target = urlsplit(self.path)
            if target.path.startswith("/evidence/"):
                try:
                    event_id = str(UUID(target.path.removeprefix("/evidence/")))
                    row = next((r for r in read_events(database) if r["event_id"] == event_id), None)
                    allowed = (database.parent / "evidence").resolve()
                    path = Path(row["evidence_path"]).resolve() if row and row["evidence_path"] else None
                    if path is None or path.parent != allowed or path.suffix.lower() != ".jpg":
                        raise ValueError
                    payload = path.read_bytes()
                except (ValueError, OSError):
                    self.send_error(404); return
                self.respond(payload, "image/jpeg"); return
            if target.path not in ("/", "/export.csv"):
                self.send_error(404); return
            params = parse_qs(target.query)
            selected = params.get("class", [""])[0]
            start = params.get("start", [""])[0]
            end = params.get("end", [""])[0]
            try:
                all_rows = read_events(database)
                rows = read_events(database, selected, start, end)
            except (ValueError, OSError, sqlite3.Error) as exc:
                self.send_error(400, str(exc)); return
            if target.path == "/export.csv":
                self.respond(csv_text(rows).encode("utf-8-sig"), "text/csv; charset=utf-8",
                             'attachment; filename="stage5h_events.csv"')
            else:
                self.respond(html_page(all_rows, rows, selected, start, end).encode("utf-8"),
                             "text/html; charset=utf-8")

        def respond(self, payload, content_type, disposition=None):
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if disposition:
                self.send_header("Content-Disposition", disposition)
            self.end_headers()
            self.wfile.write(payload)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DB)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be 1..65535")
    database = args.database.resolve()
    with ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(database)) as server:
        print(f"Local history: http://127.0.0.1:{args.port}/  database: {database}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nHistory viewer stopped.")


if __name__ == "__main__":
    main()
