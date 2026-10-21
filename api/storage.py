"""Single-database transactions for workbench state and paid-call reservations."""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path


class Store:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.root / "workbench.sqlite3", check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS entities(kind TEXT,id TEXT,body TEXT,PRIMARY KEY(kind,id))"
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS idempotency(key TEXT PRIMARY KEY,request TEXT,job_id TEXT)"
        )
        self.db.commit()

    @contextmanager
    def transaction(self):
        with self.lock:
            try:
                self.db.execute("BEGIN IMMEDIATE")
                yield
                self.db.commit()
            except BaseException:
                self.db.rollback()
                raise

    def put(self, kind, value):
        self.db.execute(
            "INSERT OR REPLACE INTO entities VALUES(?,?,?)",
            (kind, value["id"], json.dumps(value, allow_nan=False)),
        )

    def get(self, kind, ident):
        with self.lock:
            row = self.db.execute(
                "SELECT body FROM entities WHERE kind=? AND id=?", (kind, ident)
            ).fetchone()
            return json.loads(row[0]) if row else None

    def all(self, kind):
        with self.lock:
            return [
                json.loads(row[0])
                for row in self.db.execute(
                    "SELECT body FROM entities WHERE kind=? ORDER BY rowid", (kind,)
                )
            ]
