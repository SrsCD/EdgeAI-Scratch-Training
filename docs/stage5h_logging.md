# Stage 5H — local detection history

## Architecture

Stage 5G runs the existing scratch checkpoint unchanged. Every inference result
is still drawn in the live window. For logging, the highest-confidence box of
each detected class in a frame is offered to `EventStore`. The store checks the
per-class, per-camera cooldown against the latest persisted timestamp and
commits an event only when the interval has elapsed. A restart retains both
the rows and the cooldown history. This is class-level deduplication, not
object tracking: two separate backpacks in one frame produce one logged
backpack event, though both boxes remain visible.

The default database is `runs/stage5h/events.sqlite3`. Its `detection_events`
table holds UUID event ID, UTC timestamp, fixed class ID/name, model confidence,
original camera-frame `x1,y1,x2,y2` pixels, source (`webcam:N`), fixed scratch
checkpoint identifier, `model_detection_unverified` event type, optional
evidence path, notification status, and sync status. Inserts and filters use
parameterized SQL. The schema is versioned and checked on open. No database
reset occurs on application restart. The synthetic Stage 5G self-test does not
write detection events.

`--save-evidence` writes one annotated JPEG per logged event in the database's
`evidence` sibling folder. Evidence is disabled by default. Recordings are a
separate Stage 5G option. The local history server binds only to `127.0.0.1`,
uses read-only SQLite connections for queries, and serves evidence only from
that designated folder. It works without internet. CSV export applies the
selected class and UTC date filters.

## Alerts and sync

Model A contains a local SQLite logger and a future offline queue. Its source
and documentation explicitly say there is no Telegram sender or cloud sync.
No reusable Telegram credentials or configuration were found in its project
code. Model A and its database were read only.

Model B has no Telegram delivery or remote sync. Every event has
`sync_status=local_only`. Optional `--notify-class` prints a local **unverified
detection notice** for that class after a row is committed; default is no
notifications. This is a configurable class allowlist, not a threat classifier.
In particular, a backpack is never automatically labeled a security threat.
`notification_status` is `disabled` or `console_sent`; it never implies a
Telegram message. A future Telegram integration would need separately supplied
secrets, an explicit opt-in, delivery error handling, and a verified alert
policy. No token is stored in SQLite or this repository.

## Run from PowerShell

In the project folder, start live detection and logging:

```powershell
.\.venv\Scripts\python.exe scripts\stage5g_live_webcam.py --camera-index 0 --width 1280 --height 720
```

Example with per-class cooldowns, evidence, and local console notices only for
backpack and safety vest:

```powershell
.\.venv\Scripts\python.exe scripts\stage5g_live_webcam.py --camera-index 0 --cooldown-seconds 10 --cooldown-class backpack=30 --cooldown-class safety_vest=20 --save-evidence --notify-class backpack --notify-class safety_vest
```

Add `--record runs\stage5g\live_demo.mp4` to record the live annotated video.
Use `--database <path>` in both commands to select a different SQLite file.

In another PowerShell window, start the offline history viewer:

```powershell
.\.venv\Scripts\python.exe scripts\stage5h_history.py --port 8765
```

Open `http://127.0.0.1:8765/` in a browser. Filter by class or UTC date and
choose **Export filtered CSV**. Stop the viewer with Ctrl+C. The live detector
still uses Q to quit and `+`/`-` to adjust confidence.

Run the database and HTTP tests with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_stage5h.py -v
.\.venv\Scripts\python.exe scripts\stage5g_live_webcam.py --self-test
```

The database tests use labeled synthetic records and localhost HTTP requests.
They do not create genuine webcam events. The self-test runs actual inference
on a synthetic frame and does not open a physical camera.
