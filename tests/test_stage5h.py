"""Synthetic records only; these are never represented as webcam events."""
import csv
from datetime import datetime, timedelta, timezone
from io import StringIO
from http.server import ThreadingHTTPServer
from pathlib import Path
import sqlite3
import sys
import tempfile
from threading import Thread
import unittest
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from stage5h_events import EventStore, read_events
from stage5h_history import csv_text, html_page, make_handler


class EventTests(unittest.TestCase):
    def test_creation_persistence_cooldown_and_filters(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "nested/events.sqlite3"
            when = datetime(2026, 11, 24, 10, tzinfo=timezone.utc)
            with EventStore(path) as store:
                first = store.add(class_id=3, confidence=.91, xyxy=(10, 20, 90, 110),
                                  source_id="synthetic-test", cooldown=10, now=when)
                self.assertIsNotNone(first)
                self.assertIsNone(store.add(class_id=3, confidence=.95, xyxy=(11, 20, 90, 110),
                                            source_id="synthetic-test", cooldown=10,
                                            now=when + timedelta(seconds=2)))
                second = store.add(class_id=2, confidence=.8, xyxy=(1, 2, 50, 60),
                                   source_id="synthetic-test", cooldown=10, now=when)
                self.assertNotEqual(first, second)
            with EventStore(path) as reopened:
                self.assertIsNone(reopened.add(class_id=3, confidence=.9, xyxy=(10, 20, 90, 110),
                                               source_id="synthetic-test", cooldown=10,
                                               now=when + timedelta(seconds=4)))
                self.assertIsNotNone(reopened.add(class_id=3, confidence=.9, xyxy=(10, 20, 90, 110),
                                                  source_id="synthetic-test", cooldown=10,
                                                  now=when + timedelta(seconds=11)))
            rows = read_events(path)
            self.assertEqual(len(rows), 3)
            self.assertEqual(len(read_events(path, "backpack", "2026-11-24", "2026-11-24")), 2)
            self.assertEqual(len(read_events(path, "helmet")), 0)
            self.assertEqual(rows[0]["event_type"], "model_detection_unverified")
            self.assertEqual(rows[0]["sync_status"], "local_only")
            self.assertIn("Total events: 3", html_page(rows, rows, "", "", ""))
            exported = list(csv.DictReader(StringIO(csv_text(rows))))
            self.assertEqual(len(exported), 3)
            self.assertEqual({r["event_id"] for r in exported}, {r["event_id"] for r in rows})
            server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(path))
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base = f"http://127.0.0.1:{server.server_port}"
                with urlopen(base + "/?class=backpack", timeout=3) as response:
                    page = response.read().decode("utf-8")
                self.assertIn("Filtered events: 2", page)
                with urlopen(base + "/export.csv?class=backpack", timeout=3) as response:
                    exported_http = list(csv.DictReader(StringIO(response.read().decode("utf-8-sig"))))
                self.assertEqual(len(exported_http), 2)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=3)
            db = sqlite3.connect(path)
            try:
                self.assertEqual(db.execute("PRAGMA quick_check").fetchone()[0], "ok")
            finally:
                db.close()

    def test_bad_records_rejected_and_missing_db_not_created(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "events.sqlite3"
            self.assertEqual(read_events(path), [])
            self.assertFalse(path.exists())
            with EventStore(path) as store:
                for kwargs in (
                    dict(class_id=6, confidence=.8, xyxy=(0, 0, 2, 2)),
                    dict(class_id=3, confidence=float("nan"), xyxy=(0, 0, 2, 2)),
                    dict(class_id=3, confidence=.8, xyxy=(5, 0, 2, 2)),
                ):
                    with self.assertRaises(ValueError):
                        store.add(source_id="synthetic-test", cooldown=1, **kwargs)
            self.assertEqual(read_events(path), [])
            with self.assertRaises(ValueError):
                read_events(path, class_name="not_a_class")


if __name__ == "__main__":
    unittest.main()
