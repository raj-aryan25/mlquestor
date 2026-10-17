from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from mlquest.domain import (
    ApprovalRequest,
    DatasetArtifact,
    EvidenceClaim,
    EvidenceClaimType,
    EvidenceStrength,
    ExperimentSpecification,
    Hypothesis,
    MetricDirection,
    MetricResult,
    PhaseState,
    ProviderCallRecord,
    ProviderCallStatus,
    ProviderUsage,
    ResearchRecord,
    RunAttempt,
    RunStatus,
    SourceType,
    SplitSpecification,
    SplitStrategy,
    ValidationSpecification,
    VerificationResult,
    VerificationStatus,
)
from mlquest.persistence import CURRENT_SCHEMA_VERSION, LocalLedger


def test_schema_creation_enables_wal_and_named_tables(tmp_path) -> None:
    path = tmp_path / "ledger.sqlite3"
    ledger = LocalLedger(path)
    assert ledger.journal_mode == "wal"
    assert ledger.schema_version == CURRENT_SCHEMA_VERSION
    tables = {
        row[0]
        for row in ledger._connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    assert {"sessions", "phases", "events", "run_attempts", "verification"} <= tables
    ledger.close()


def test_session_and_phase_survive_restart(tmp_path) -> None:
    path = tmp_path / "ledger.sqlite3"
    with LocalLedger(path) as ledger:
        ledger.create_session("session-1", {"name": "demo"})
        ledger.save_phase("session-1", PhaseState(id="phase-1", phase="dataset", status="running"))
    with LocalLedger(path) as ledger:
        assert ledger.get_session("session-1") == {"name": "demo"}
        phase = ledger.get_phase("session-1", "dataset")
        assert phase is not None
        assert phase["status"] == "running"


def test_events_are_append_only_and_ordered(tmp_path) -> None:
    with LocalLedger(tmp_path / "ledger.sqlite3") as ledger:
        first = ledger.append_event("session-1", "phase.started", {"phase": "intake"})
        second = ledger.append_event(
            "session-1",
            "run.queued",
            {"attempt": 1},
            entity_type="run_attempt",
            entity_id="run-1",
        )
        events = ledger.list_events("session-1")
    assert [event.sequence for event in events] == [first, second]
    assert events[1].entity_id == "run-1"
    assert events[0].payload == {"phase": "intake"}


def test_domain_records_round_trip_as_json_and_can_be_rehydrated(tmp_path) -> None:
    validation = ValidationSpecification(
        split=SplitSpecification(strategy=SplitStrategy.RANDOM, test_size=0.2, random_seed=4),
        primary_metric="accuracy",
        metric_direction=MetricDirection.MAXIMIZE,
    )
    experiment = ExperimentSpecification(
        id="experiment-1", name="baseline", model_family="linear", validation=validation
    )
    with LocalLedger(tmp_path / "ledger.sqlite3") as ledger:
        ledger.put_record("experiments", experiment, session_id="session-1")
        raw = ledger.get_record("experiments", experiment.id)
        restored = ledger.get_record("experiments", experiment.id, ExperimentSpecification)
    assert raw == experiment.model_dump(mode="json", exclude_none=True)
    assert restored == experiment


def test_provider_call_metadata_round_trips_without_bodies(tmp_path) -> None:
    record = ProviderCallRecord(
        id="provider-call-1",
        operation="interpret_task",
        provider="openrouter",
        model="test/model",
        status=ProviderCallStatus.SUCCEEDED,
        response_schema="mlquest.domain.planning.TaskInterpretation",
        prompt_digest="a" * 64,
        prompt_chars=128,
        response_digest="b" * 64,
        response_chars=256,
        usage=ProviderUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        attempt_count=2,
        started_at="2026-01-01T00:00:00Z",
        finished_at="2026-01-01T00:00:01Z",
    )
    with LocalLedger(tmp_path / "ledger.sqlite") as ledger:
        ledger.put_record("provider_calls", record, session_id="session-1")
        restored = ledger.get_record("provider_calls", record.id, ProviderCallRecord)
        payload = ledger._connection.execute(
            "SELECT record_json FROM provider_calls WHERE id = ?", (record.id,)
        ).fetchone()[0]

    assert restored == record
    assert "prompt_chars" in payload
    assert "prompt body" not in payload


def test_all_step_three_record_categories_use_json_payloads(tmp_path) -> None:
    dataset = DatasetArtifact(
        id="dataset-1",
        role="train",
        content_hash="a" * 64,
        media_type="text/csv",
        immutable_location="artifacts/train.csv",
        format="csv",
    )
    records = {
        "approvals": ApprovalRequest(
            id="approval-1", phase="experiment", requested_action="run", rationale="baseline"
        ),
        "datasets": dataset,
        "artifacts": dataset,
        "research": ResearchRecord(
            id="research-1",
            title="Paper",
            url="https://example.test/paper",
            source_type=SourceType.PAPER,
            accessed_at="2026-01-01T00:00:00Z",
        ),
        "hypotheses": Hypothesis(
            id="hypothesis-1", statement="feature helps", rationale="matching evidence"
        ),
        "metrics": MetricResult(
            id="metric-1",
            metric_name="accuracy",
            value=0.8,
            direction=MetricDirection.MAXIMIZE,
            run_attempt_id="run-1",
            comparison_context_id="context-1",
        ),
        "claims": EvidenceClaim(
            id="claim-1",
            claim="the result improved",
            claim_type=EvidenceClaimType.MEASURED_RESULT,
            strength=EvidenceStrength.PROVISIONAL,
        ),
        "verification": VerificationResult(
            id="verification-1", run_attempt_id="run-1", status=VerificationStatus.INCOMPLETE
        ),
    }
    with LocalLedger(tmp_path / "ledger.sqlite3") as ledger:
        for category, record in records.items():
            ledger.put_record(category, record, session_id="session-1")
        loaded = {
            category: ledger.get_record(category, record.id) for category, record in records.items()
        }
    assert loaded == {
        category: record.model_dump(mode="json", exclude_none=True)
        for category, record in records.items()
    }


def test_migration_version_is_recorded_and_reused(tmp_path) -> None:
    path = tmp_path / "ledger.sqlite3"
    with LocalLedger(path) as ledger:
        assert ledger.schema_version == CURRENT_SCHEMA_VERSION
        migrations = ledger._connection.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()
    assert [row[0] for row in migrations] == list(range(1, CURRENT_SCHEMA_VERSION + 1))
    connection = sqlite3.connect(path)
    assert connection.execute("PRAGMA user_version").fetchone()[0] == CURRENT_SCHEMA_VERSION
    connection.close()


def test_existing_v1_database_receives_new_indexes(tmp_path) -> None:
    path = tmp_path / "ledger.sqlite3"
    with LocalLedger(path) as ledger:
        ledger._connection.execute("DROP INDEX idx_events_session_sequence")
        ledger._connection.execute("DELETE FROM schema_migrations WHERE version = 2")
        ledger._connection.execute("PRAGMA user_version = 1")
        ledger._connection.commit()
    with LocalLedger(path) as ledger:
        index = ledger._connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index' "
            "AND name = 'idx_events_session_sequence'"
        ).fetchone()
        assert ledger.schema_version == CURRENT_SCHEMA_VERSION
    assert index is not None


def test_existing_v3_database_receives_provider_call_table(tmp_path) -> None:
    path = tmp_path / "ledger.sqlite3"
    with LocalLedger(path) as ledger:
        ledger._connection.execute("DROP TABLE provider_calls")
        ledger._connection.execute("DELETE FROM schema_migrations WHERE version = 4")
        ledger._connection.execute("PRAGMA user_version = 3")
        ledger._connection.commit()
    with LocalLedger(path) as ledger:
        table = ledger._connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'provider_calls'"
        ).fetchone()
        assert ledger.schema_version == CURRENT_SCHEMA_VERSION
    assert table is not None


def test_incomplete_phases_and_in_flight_runs_are_queryable(tmp_path) -> None:
    with LocalLedger(tmp_path / "ledger.sqlite3") as ledger:
        ledger.create_session("session-1")
        ledger.save_phase("session-1", {"id": "phase-a", "phase": "experiment", "status": "paused"})
        ledger.save_phase("session-1", {"id": "phase-b", "phase": "dataset", "status": "completed"})
        queued = RunAttempt(id="run-queued", experiment_id="experiment-1", status=RunStatus.QUEUED)
        running = RunAttempt(
            id="run-running",
            experiment_id="experiment-1",
            status=RunStatus.RUNNING,
            started_at=datetime.now(timezone.utc),
        )
        finished = RunAttempt(
            id="run-finished",
            experiment_id="experiment-1",
            status=RunStatus.FAILED,
            failure_reason="fixture failure",
        )
        for run in (queued, running, finished):
            ledger.put_record("run_attempts", run, session_id="session-1")
        phases = ledger.incomplete_phases("session-1")
        runs = ledger.in_flight_runs("session-1")
    assert [phase["id"] for phase in phases] == ["phase-a"]
    assert {run["id"] for run in runs} == {"run-queued", "run-running"}
