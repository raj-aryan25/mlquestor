"""SQLite ledger for resumable sessions and evidence records.

The ledger stores small structured records as canonical JSON. Files referenced by
artifact manifests remain outside the database and are intentionally not handled
by this module.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, TypeVar, cast, overload
from uuid import uuid4

from pydantic import BaseModel

CURRENT_SCHEMA_VERSION = 4
_T = TypeVar("_T", bound=BaseModel)


class LedgerError(RuntimeError):
    """Base exception for ledger failures."""


class MigrationError(LedgerError):
    """Raised when a database cannot be migrated safely."""


@dataclass(frozen=True, slots=True)
class StoredEvent:
    """An append-only event returned by :meth:`LocalLedger.list_events`."""

    sequence: int
    session_id: str
    event_type: str
    occurred_at: str
    payload: dict[str, Any]
    entity_type: str | None = None
    entity_id: str | None = None


_CATEGORY_TABLES = {
    "session": "sessions",
    "sessions": "sessions",
    "phase": "phases",
    "phases": "phases",
    "approval": "approvals",
    "approvals": "approvals",
    "dataset": "datasets",
    "datasets": "datasets",
    "artifact": "artifacts",
    "artifacts": "artifacts",
    "research": "research",
    "hypothesis": "hypotheses",
    "hypotheses": "hypotheses",
    "experiment": "experiments",
    "experiments": "experiments",
    "run_attempt": "run_attempts",
    "run_attempts": "run_attempts",
    "run": "run_attempts",
    "metric": "metrics",
    "metrics": "metrics",
    "claim": "claims",
    "claims": "claims",
    "verification": "verification",
    "verifications": "verification",
    "verification_run": "verification",
    "score": "external_scores",
    "scores": "external_scores",
    "external_score": "external_scores",
    "external_scores": "external_scores",
    "external_score_observation": "external_scores",
    "external_score_observations": "external_scores",
    "provider_call": "provider_calls",
    "provider_calls": "provider_calls",
}

_STATUS_FIELDS = {
    "phases",
    "approvals",
    "hypotheses",
    "experiments",
    "run_attempts",
    "verification",
    "provider_calls",
}


def _json_ready(value: Any) -> Any:
    """Convert supported values to JSON primitives without lossy repr()."""

    if isinstance(value, Enum):
        return _json_ready(value.value)
    if isinstance(value, datetime):
        timestamp = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        return timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "model_dump"):
        return _json_ready(value.model_dump(mode="json", exclude_none=True))
    raise TypeError(f"unsupported value for JSON persistence: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Return deterministic, compact JSON for a supported record or mapping."""

    if hasattr(value, "canonical_json"):
        value = json.loads(value.canonical_json())
    elif hasattr(value, "model_dump"):
        value = value.model_dump(mode="json", exclude_none=True)
    return json.dumps(
        _json_ready(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _record_id(record: Any) -> str | None:
    if isinstance(record, Mapping):
        value = record.get("id")
    else:
        value = getattr(record, "id", None)
    return str(value) if value is not None else None


def _record_value(record: Any, name: str) -> Any:
    if isinstance(record, Mapping):
        return record.get(name)
    return getattr(record, name, None)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _payload_from_json(value: str) -> dict[str, Any]:
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise LedgerError("persisted record payload must be a JSON object")
    return payload


_BASE_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    record_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS phases (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    phase TEXT NOT NULL,
    status TEXT NOT NULL,
    attempt INTEGER NOT NULL DEFAULT 0,
    record_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(session_id, phase)
);

CREATE TABLE IF NOT EXISTS approvals (
    id TEXT PRIMARY KEY,
    session_id TEXT,
    phase TEXT,
    status TEXT,
    record_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS artifacts (
    id TEXT PRIMARY KEY,
    session_id TEXT,
    kind TEXT,
    content_hash TEXT,
    record_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS datasets (
    id TEXT PRIMARY KEY,
    session_id TEXT,
    role TEXT,
    record_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS research (
    id TEXT PRIMARY KEY,
    session_id TEXT,
    record_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS hypotheses (
    id TEXT PRIMARY KEY,
    session_id TEXT,
    status TEXT,
    record_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    session_id TEXT,
    status TEXT,
    record_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS run_attempts (
    id TEXT PRIMARY KEY,
    session_id TEXT,
    experiment_id TEXT,
    status TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    record_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS metrics (
    id TEXT PRIMARY KEY,
    session_id TEXT,
    run_attempt_id TEXT,
    record_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS claims (
    id TEXT PRIMARY KEY,
    session_id TEXT,
    record_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS verification (
    id TEXT PRIMARY KEY,
    session_id TEXT,
    run_attempt_id TEXT,
    status TEXT,
    record_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    entity_type TEXT,
    entity_id TEXT,
    occurred_at TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
"""

_INDEX_SCHEMA = """
CREATE INDEX IF NOT EXISTS idx_phases_session_status ON phases(session_id, status);
CREATE INDEX IF NOT EXISTS idx_approvals_session_status ON approvals(session_id, status);
CREATE INDEX IF NOT EXISTS idx_run_attempts_session_status ON run_attempts(session_id, status);
CREATE INDEX IF NOT EXISTS idx_run_attempts_experiment ON run_attempts(experiment_id);
CREATE INDEX IF NOT EXISTS idx_events_session_sequence ON events(session_id, sequence);
"""

_SCORE_SCHEMA = """
CREATE TABLE IF NOT EXISTS external_scores (
    id TEXT PRIMARY KEY,
    session_id TEXT,
    record_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_external_scores_session ON external_scores(session_id, created_at);
"""

_PROVIDER_SCHEMA = """
CREATE TABLE IF NOT EXISTS provider_calls (
    id TEXT PRIMARY KEY,
    session_id TEXT,
    provider TEXT,
    model TEXT,
    status TEXT,
    record_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_provider_calls_session ON provider_calls(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_provider_calls_provider_model
    ON provider_calls(provider, model, status);
"""


class LocalLedger:
    """A small, commit-on-write SQLite ledger for one local MLQuest workspace."""

    def __init__(self, path: str | Path, *, timeout: float = 30.0) -> None:
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(
            ":memory:" if str(path) == ":memory:" else self.path,
            timeout=timeout,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA busy_timeout = 30000")
        self.journal_mode = str(
            self._connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]
        ).lower()
        self._connection.execute("PRAGMA synchronous = FULL")
        self._migrate()

    @property
    def schema_version(self) -> int:
        return int(self._connection.execute("PRAGMA user_version").fetchone()[0])

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "LocalLedger":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            yield
        except Exception:
            self._connection.rollback()
            raise
        else:
            self._connection.commit()

    def _migrate(self) -> None:
        self._connection.executescript(
            "CREATE TABLE IF NOT EXISTS schema_migrations "
            "(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL);"
        )
        current = self.schema_version
        if current > CURRENT_SCHEMA_VERSION:
            raise MigrationError(
                f"database schema version {current} is newer than supported version "
                f"{CURRENT_SCHEMA_VERSION}"
            )
        migrations = {1: _BASE_SCHEMA, 2: _INDEX_SCHEMA, 3: _SCORE_SCHEMA, 4: _PROVIDER_SCHEMA}
        for version in range(current + 1, CURRENT_SCHEMA_VERSION + 1):
            try:
                with self._transaction():
                    self._connection.executescript(migrations[version])
                    self._connection.execute(
                        "INSERT OR REPLACE INTO schema_migrations("
                        "version, applied_at) VALUES (?, ?)",
                        (version, _utc_now()),
                    )
                    self._connection.execute(f"PRAGMA user_version = {version}")
            except sqlite3.Error as exc:
                raise MigrationError(f"failed to apply schema migration {version}") from exc

    def create_session(
        self,
        session_id: str | None = None,
        data: Any = None,
        *,
        status: str = "active",
    ) -> str:
        identifier = session_id or str(uuid4())
        self.save_session(identifier, data, status=status)
        return identifier

    def save_session(
        self, session_id: str | Any, data: Any = None, *, status: str = "active"
    ) -> str:
        if not isinstance(session_id, str):
            record = session_id
            identifier = _record_id(record) or str(uuid4())
            data = record if data is None else data
        else:
            identifier = session_id
        payload = canonical_json({} if data is None else data)
        now = _utc_now()
        with self._transaction():
            self._connection.execute(
                """INSERT INTO sessions(id, status, record_json, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET status=excluded.status,
                     record_json=excluded.record_json, updated_at=excluded.updated_at""",
                (identifier, status, payload, now, now),
            )
        return identifier

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        row = self._connection.execute(
            "SELECT record_json FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return None if row is None else _payload_from_json(row["record_json"])

    def get_session_metadata(self, session_id: str) -> dict[str, Any] | None:
        row = self._connection.execute(
            """SELECT id, status, created_at, updated_at, record_json
               FROM sessions WHERE id = ?""",
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "status": row["status"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "record": _payload_from_json(row["record_json"]),
        }

    def save_phase(
        self,
        session_id: str,
        phase: Any,
        data: Any = None,
        *,
        status: str | None = None,
        attempt: int | None = None,
    ) -> str:
        phase_name = str(_record_value(phase, "phase") or phase)
        payload_source = (
            phase if data is None and not isinstance(phase, str) else ({} if data is None else data)
        )
        payload = canonical_json(payload_source)
        record = json.loads(payload)
        phase_status = status or str(record.get("status", "pending"))
        phase_attempt = attempt if attempt is not None else int(record.get("attempt", 0))
        identifier = _record_id(phase) or f"{session_id}:{phase_name}"
        now = _utc_now()
        with self._transaction():
            self._connection.execute(
                """INSERT INTO phases(
                   id, session_id, phase, status, attempt, record_json,
                   created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET session_id=excluded.session_id,
                     phase=excluded.phase, status=excluded.status, attempt=excluded.attempt,
                     record_json=excluded.record_json, updated_at=excluded.updated_at""",
                (
                    identifier,
                    session_id,
                    phase_name,
                    phase_status,
                    phase_attempt,
                    payload,
                    now,
                    now,
                ),
            )
        return identifier

    def get_phase(self, session_id: str, phase: str) -> dict[str, Any] | None:
        row = self._connection.execute(
            "SELECT record_json FROM phases WHERE session_id = ? AND phase = ?",
            (session_id, phase),
        ).fetchone()
        return None if row is None else _payload_from_json(row["record_json"])

    def put_record(self, category: str, record: Any, *, session_id: str | None = None) -> str:
        """Insert or replace a JSON record in one of the Step 3 categories."""

        table = _CATEGORY_TABLES.get(category.lower())
        if table is None:
            raise ValueError(f"unsupported ledger record category: {category}")
        identifier = _record_id(record)
        if not identifier:
            raise ValueError("persisted records require an id")
        payload = canonical_json(record)
        values = json.loads(payload)
        stored_session = session_id or _record_value(record, "session_id")
        now = _utc_now()
        columns = ["id", "session_id", "record_json", "created_at", "updated_at"]
        params: list[Any] = [identifier, stored_session, payload, now, now]
        if table == "sessions":
            columns = ["id", "status", "record_json", "created_at", "updated_at"]
            params = [identifier, str(values.get("status", "active")), payload, now, now]
        else:
            extra: dict[str, Any] = {}
            if table == "phases":
                extra = {
                    "phase": values.get("phase", ""),
                    "status": values.get("status", "pending"),
                    "attempt": values.get("attempt", 0),
                }
            elif table == "approvals":
                extra = {"phase": values.get("phase"), "status": values.get("status")}
            elif table == "artifacts":
                extra = {
                    "kind": values.get("kind"),
                    "content_hash": values.get("content_hash", values.get("sha256")),
                }
            elif table == "datasets":
                extra = {"role": values.get("role")}
            elif table == "provider_calls":
                extra = {
                    "provider": values.get("provider"),
                    "model": values.get("model"),
                    "status": values.get("status"),
                }
            elif table in _STATUS_FIELDS:
                extra = {"status": values.get("status")}
            if table == "run_attempts":
                extra.update(
                    {
                        "experiment_id": values.get("experiment_id"),
                        "started_at": values.get("started_at"),
                        "finished_at": values.get("finished_at"),
                    }
                )
            if table == "metrics":
                extra["run_attempt_id"] = values.get("run_attempt_id")
            if table == "verification":
                extra["run_attempt_id"] = values.get("run_attempt_id")
            columns = ["id", "session_id", *extra, "record_json", "created_at", "updated_at"]
            params = [identifier, stored_session, *extra.values(), payload, now, now]
        placeholders = ", ".join("?" for _ in columns)
        assignments = ", ".join(
            f"{column}=excluded.{column}" for column in columns[1:] if column != "created_at"
        )
        sql = (
            f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders}) "
            f"ON CONFLICT(id) DO UPDATE SET {assignments}"
        )
        with self._transaction():
            self._connection.execute(sql, params)
        return identifier

    @overload
    def get_record(
        self,
        category: str,
        record_id: str,
        model: None = None,
    ) -> dict[str, Any] | None: ...

    @overload
    def get_record(
        self,
        category: str,
        record_id: str,
        model: type[_T],
    ) -> _T | None: ...

    def get_record(
        self,
        category: str,
        record_id: str,
        model: type[_T] | None = None,
    ) -> dict[str, Any] | _T | None:
        table = _CATEGORY_TABLES.get(category.lower())
        if table is None:
            raise ValueError(f"unsupported ledger record category: {category}")
        row = self._connection.execute(
            f"SELECT record_json FROM {table} WHERE id = ?", (record_id,)
        ).fetchone()
        if row is None:
            return None
        payload = _payload_from_json(row["record_json"])
        return payload if model is None else model.model_validate(payload)

    @overload
    def list_records(
        self,
        category: str,
        *,
        session_id: str | None = None,
        model: None = None,
    ) -> list[dict[str, Any]]: ...

    @overload
    def list_records(
        self,
        category: str,
        *,
        session_id: str | None = None,
        model: type[_T],
    ) -> list[_T]: ...

    def list_records(
        self,
        category: str,
        *,
        session_id: str | None = None,
        model: type[_T] | None = None,
    ) -> list[Any]:
        table = _CATEGORY_TABLES.get(category.lower())
        if table is None:
            raise ValueError(f"unsupported ledger record category: {category}")
        if table == "sessions":
            query = "SELECT record_json FROM sessions"
            params: tuple[Any, ...] = ()
        elif session_id is None:
            query = f"SELECT record_json FROM {table} ORDER BY created_at, id"
            params = ()
        else:
            query = f"SELECT record_json FROM {table} WHERE session_id = ? ORDER BY created_at, id"
            params = (session_id,)
        rows = self._connection.execute(query, params).fetchall()
        payloads = [_payload_from_json(row["record_json"]) for row in rows]
        return (
            payloads if model is None else [model.model_validate(payload) for payload in payloads]
        )

    def append_event(
        self,
        session_id: str,
        event_type: str,
        payload: Any = None,
        *,
        entity_type: str | None = None,
        entity_id: str | None = None,
        occurred_at: datetime | str | None = None,
    ) -> int:
        if not event_type.strip():
            raise ValueError("event_type must not be empty")
        timestamp = occurred_at
        if isinstance(timestamp, datetime):
            timestamp = _json_ready(timestamp)
        timestamp = timestamp or _utc_now()
        event_payload = canonical_json({} if payload is None else payload)
        with self._transaction():
            cursor = self._connection.execute(
                """INSERT INTO events(
                   session_id, event_type, entity_type, entity_id, occurred_at,
                   payload_json)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (session_id, event_type, entity_type, entity_id, timestamp, event_payload),
            )
            return int(cast(int, cursor.lastrowid))

    def list_events(self, session_id: str, *, after_sequence: int = 0) -> list[StoredEvent]:
        rows = self._connection.execute(
            """SELECT sequence, session_id, event_type, entity_type, entity_id,
               occurred_at, payload_json
               FROM events WHERE session_id = ? AND sequence > ? ORDER BY sequence""",
            (session_id, after_sequence),
        ).fetchall()
        return [
            StoredEvent(
                sequence=int(row["sequence"]),
                session_id=str(row["session_id"]),
                event_type=str(row["event_type"]),
                occurred_at=str(row["occurred_at"]),
                payload=_payload_from_json(row["payload_json"]),
                entity_type=row["entity_type"],
                entity_id=row["entity_id"],
            )
            for row in rows
        ]

    def list_incomplete_phases(self, session_id: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT record_json FROM phases WHERE status NOT IN ('completed', 'rejected')"
        params: tuple[Any, ...] = ()
        if session_id is not None:
            query += " AND session_id = ?"
            params = (session_id,)
        query += " ORDER BY updated_at, id"
        return [
            _payload_from_json(row["record_json"])
            for row in self._connection.execute(query, params)
        ]

    def list_in_flight_runs(self, session_id: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT record_json FROM run_attempts WHERE status IN ('queued', 'running')"
        params: tuple[Any, ...] = ()
        if session_id is not None:
            query += " AND session_id = ?"
            params = (session_id,)
        query += " ORDER BY updated_at, id"
        return [
            _payload_from_json(row["record_json"])
            for row in self._connection.execute(query, params)
        ]

    incomplete_phases = list_incomplete_phases
    in_flight_runs = list_in_flight_runs
    save_record = put_record
    load_record = get_record
    get_incomplete_phases = list_incomplete_phases
    get_in_flight_runs = list_in_flight_runs
    resume_incomplete_phases = list_incomplete_phases
    resume_in_flight_runs = list_in_flight_runs


SQLiteLedger = LocalLedger
