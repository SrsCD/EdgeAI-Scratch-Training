"""Local, unverified detection events. No network or cloud delivery."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
from pathlib import Path
import sqlite3
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "runs/stage5h/events.sqlite3"
NAMES = ("person", "helmet", "safety_vest", "backpack", "knife", "pistol")
CHECKPOINT_ID = "scratch_yolo11n_stage3c_experimental_b16/best.pt:1bb3228630eb609b"

SCHEMA = """
CREATE TABLE IF NOT EXISTS detection_events (
 event_id TEXT PRIMARY KEY,
 timestamp_utc TEXT NOT NULL,
 class_id INTEGER NOT NULL CHECK(class_id BETWEEN 0 AND 5),
 class_name TEXT NOT NULL,
 confidence REAL NOT NULL CHECK(confidence >= 0 AND confidence <= 1),
 x1 REAL NOT NULL, y1 REAL NOT NULL, x2 REAL NOT NULL, y2 REAL NOT NULL,
 source_id TEXT NOT NULL,
 checkpoint_id TEXT NOT NULL,
 event_type TEXT NOT NULL CHECK(event_type = 'model_detection_unverified'),
 evidence_path TEXT,
 notification_status TEXT NOT NULL CHECK(notification_status IN ('disabled','console_sent')),
 sync_status TEXT NOT NULL CHECK(sync_status = 'local_only')
);
CREATE INDEX IF NOT EXISTS idx_events_time ON detection_events(timestamp_utc);
CREATE INDEX IF NOT EXISTS idx_events_class_time ON detection_events(class_id,timestamp_utc);
"""


def validate_event(class_id, confidence, xyxy, source_id):
    if type(class_id) is not int or not 0 <= class_id < 6:
        raise ValueError("Invalid class ID")
    if not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("Invalid confidence")
    if len(xyxy) != 4 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in xyxy):
        raise ValueError("Invalid bounding box")
    x1, y1, x2, y2 = xyxy
    if not (0 <= x1 < x2 and 0 <= y1 < y2):
        raise ValueError("Bounding box must have positive area")
    if not source_id or not isinstance(source_id, str):
        raise ValueError("Source identifier is required")


class EventStore:
    def __init__(self, path: Path = DEFAULT_DB, checkpoint_id: str = CHECKPOINT_ID):
        self.path = Path(path).resolve()
        self.checkpoint_id = checkpoint_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=5)
        self.db.row_factory = sqlite3.Row
        try:
            self.db.execute("PRAGMA busy_timeout=5000")
            version = self.db.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1):
                raise RuntimeError(f"Unsupported event database schema version {version}")
            exists = self.db.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='detection_events'").fetchone()
            if exists:
                columns = {row[1] for row in self.db.execute("PRAGMA table_info(detection_events)")}
                required = {"event_id", "timestamp_utc", "class_id", "class_name", "confidence",
                            "x1", "y1", "x2", "y2", "source_id", "checkpoint_id",
                            "event_type", "evidence_path", "notification_status", "sync_status"}
                if not required <= columns:
                    raise RuntimeError("Existing detection_events table has an incompatible schema")
            self.db.executescript(SCHEMA)
            self.db.execute("PRAGMA user_version=1")
            self.db.commit()
        except Exception:
            self.db.close()
            raise

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def add(self, *, class_id: int, confidence: float, xyxy, source_id: str,
            cooldown: float, evidence_path: str | None = None,
            notification_status: str = "disabled", now: datetime | None = None) -> str | None:
        validate_event(class_id, confidence, xyxy, source_id)
        if not math.isfinite(cooldown) or cooldown < 0:
            raise ValueError("Cooldown must be finite and nonnegative")
        if notification_status not in ("disabled", "console_sent"):
            raise ValueError("Invalid notification status")
        instant = now or datetime.now(timezone.utc)
        if instant.tzinfo is None or instant.utcoffset() is None:
            raise ValueError("Timestamp must include a timezone")
        instant = instant.astimezone(timezone.utc)
        event_id = str(uuid4())
        with self.db:
            # SQLite serialization plus persisted timestamps makes cooldown
            # effective across process restarts. One event per class/source.
            previous = self.db.execute(
                "SELECT timestamp_utc FROM detection_events WHERE class_id=? AND source_id=? "
                "ORDER BY timestamp_utc DESC LIMIT 1", (class_id, source_id)).fetchone()
            if previous and (instant - datetime.fromisoformat(previous[0])).total_seconds() < cooldown:
                return None
            self.db.execute("""INSERT INTO detection_events
                (event_id,timestamp_utc,class_id,class_name,confidence,x1,y1,x2,y2,
                 source_id,checkpoint_id,event_type,evidence_path,notification_status,sync_status)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,'model_detection_unverified',?,?,'local_only')""",
                (event_id, instant.isoformat(timespec="milliseconds"), class_id, NAMES[class_id],
                 float(confidence), *map(float, xyxy), source_id, self.checkpoint_id,
                 evidence_path, notification_status))
        return event_id

    def set_evidence(self, event_id: str, path: Path):
        with self.db:
            self.db.execute("UPDATE detection_events SET evidence_path=? WHERE event_id=?",
                            (str(path.resolve()), event_id))


def read_events(path: Path, class_name: str = "", start: str = "", end: str = ""):
    """Read-only filtered rows and counts; date bounds are YYYY-MM-DD UTC."""
    if class_name and class_name not in NAMES:
        raise ValueError("Unknown class filter")
    for value in (start, end):
        if value:
            datetime.strptime(value, "%Y-%m-%d")
    path = Path(path).resolve()
    if not path.is_file():
        return []
    db = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=5)
    db.row_factory = sqlite3.Row
    try:
        db.execute("PRAGMA query_only=ON")
        where, params = [], []
        if class_name:
            where.append("class_name=?"); params.append(class_name)
        if start:
            where.append("timestamp_utc>=?"); params.append(start + "T")
        if end:
            next_day = datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1)
            where.append("timestamp_utc<?"); params.append(next_day.strftime("%Y-%m-%dT"))
        sql = "SELECT * FROM detection_events" + (" WHERE " + " AND ".join(where) if where else "")
        return [dict(row) for row in db.execute(sql + " ORDER BY timestamp_utc DESC", params)]
    finally:
        db.close()
