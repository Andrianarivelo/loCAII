"""Durable user steering must survive interruption and enter history exactly once."""

from concurrent.futures import ThreadPoolExecutor
import sqlite3

import pytest

from locaii.storage import Store


@pytest.fixture
def store(tmp_path):
    repository = Store(tmp_path / "state")
    yield repository
    repository.close()


def session(store, tmp_path, title="Synthetic task"):
    """Create a private fixture task without running an inference worker."""
    return store.create_session(title, tmp_path / "workspace")["id"]


def test_queue_preserves_exact_text_and_delivers_once(store, tmp_path):
    identifier = session(store, tmp_path)
    store.append(identifier, "user", "Original objective")
    first = store.queue_steering(identifier, "  Please preserve exact spacing.\n")
    second = store.queue_steering(identifier, "Use the blue option.")
    assert store.pending_steering(identifier) == [first, second]
    assert store.messages(identifier) == [
        {"role": "user", "content": "Original objective"}
    ]
    delivered = store.consume_steering(identifier)
    assert delivered == [
        {"role": "user", "content": first["text"], "_steering_id": first["id"]},
        {"role": "user", "content": second["text"], "_steering_id": second["id"]},
    ]
    assert store.messages(identifier)[1:] == delivered
    assert store.pending_steering(identifier) == []
    assert store.consume_steering(identifier) == []
    assert store.messages(identifier)[1:] == delivered
    assert [
        event["id"]
        for event in store.events(identifier)
        if event["type"] == "steering_queued"
    ] == [first["id"], second["id"]]


def test_pending_survives_cancellation_restart_and_session_isolation(tmp_path):
    root = tmp_path / "state"
    first_store = Store(root)
    identifier = session(first_store, tmp_path)
    other = session(first_store, tmp_path, "Other task")
    pending = first_store.queue_steering(
        identifier, "Resume with this exact constraint."
    )
    other_pending = first_store.queue_steering(other, "Other constraint")
    first_store.set_status(identifier, "interrupted")
    assert first_store.pending_steering(identifier) == [pending]
    first_store.close()
    restarted = Store(root)
    try:
        assert restarted.pending_steering(identifier) == [pending]
        assert restarted.messages(identifier) == []
        assert restarted.get_session(identifier)["status"] == "interrupted"
        restarted.consume_steering(identifier)
        assert restarted.pending_steering(other) == [other_pending]
        assert restarted.messages(other) == []
    finally:
        restarted.close()
    again = Store(root)
    try:
        assert again.consume_steering(identifier) == []
        assert len(again.messages(identifier)) == 1
    finally:
        again.close()


def test_concurrent_queue_and_consume_are_ordered_and_exactly_once(store, tmp_path):
    identifier = session(store, tmp_path)
    with ThreadPoolExecutor(max_workers=8) as executor:
        submitted = list(
            executor.map(
                lambda index: store.queue_steering(identifier, f"Direction {index}"),
                range(24),
            )
        )
    expected = sorted(submitted, key=lambda item: item["id"])
    assert store.pending_steering(identifier) == expected
    with ThreadPoolExecutor(max_workers=8) as executor:
        consumed = list(
            executor.map(lambda _: store.consume_steering(identifier), range(8))
        )
    delivered = [item for group in consumed for item in group]
    assert len(delivered) == 24
    assert len({item["_steering_id"] for item in delivered}) == 24
    assert [item["_steering_id"] for item in store.messages(identifier)] == [
        item["id"] for item in expected
    ]


def test_atomic_rollback_keeps_messages_and_pending_consistent(store, tmp_path):
    identifier = session(store, tmp_path)
    queued = [
        store.queue_steering(identifier, f"Direction {number}") for number in range(3)
    ]
    # Force failure after the first message insertion. A partial consume must
    # never leave a delivered message beside its still-pending queue receipt.
    store.db.execute(
        "CREATE TRIGGER fail_steering_update BEFORE UPDATE ON steering BEGIN SELECT RAISE(ABORT, 'synthetic delivery failure'); END"
    )
    store.db.commit()
    with pytest.raises(sqlite3.IntegrityError, match="synthetic"):
        store.consume_steering(identifier)
    assert store.messages(identifier) == []
    assert store.pending_steering(identifier) == queued
    store.db.execute("DROP TRIGGER fail_steering_update")
    store.db.commit()
    assert len(store.consume_steering(identifier)) == 3
    assert store.pending_steering(identifier) == []


def test_queue_cap_text_validation_and_unknown_sessions(store, tmp_path):
    identifier = session(store, tmp_path)
    for bad in (None, 3, [], "", " \n ", "x" * 32001):
        with pytest.raises(ValueError):
            store.queue_steering(identifier, bad)
    accepted = store.queue_steering(identifier, "x" * 32000)
    assert len(accepted["text"]) == 32000
    for index in range(31):
        store.queue_steering(identifier, f"Direction {index}")
    with pytest.raises(ValueError, match="32 pending"):
        store.queue_steering(identifier, "Overflow")
    assert len(store.pending_steering(identifier)) == 32
    assert len(store.consume_steering(identifier)) == 32
    assert (
        store.queue_steering(identifier, "Room after delivery")["status"] == "pending"
    )
    for unknown in ("missing", None, []):
        with pytest.raises(KeyError):
            store.queue_steering(unknown, "Direction")
        with pytest.raises(KeyError):
            store.pending_steering(unknown)
        with pytest.raises(KeyError):
            store.consume_steering(unknown)


def test_queue_event_failure_rolls_back_pending_receipt(store, tmp_path):
    identifier = session(store, tmp_path)
    store.db.execute(
        "CREATE TRIGGER fail_steering_event BEFORE INSERT ON events BEGIN SELECT RAISE(ABORT, 'synthetic event failure'); END"
    )
    store.db.commit()
    with pytest.raises(sqlite3.IntegrityError):
        store.queue_steering(identifier, "Must not be partly acknowledged")
    assert store.pending_steering(identifier) == []
    assert store.events(identifier) == []


def test_independent_connections_share_queue_cap_and_atomic_claims(tmp_path):
    """SQLite write locking protects the queue beyond one Store instance."""
    root = tmp_path / "state"
    first = Store(root)
    identifier = session(first, tmp_path)
    second = Store(root)

    def enqueue(index):
        try:
            return (first if index % 2 else second).queue_steering(
                identifier, f"Direction {index}"
            )
        except ValueError as error:
            assert "32 pending" in str(error)
            return None

    try:
        with ThreadPoolExecutor(max_workers=8) as executor:
            submitted = list(executor.map(enqueue, range(40)))
        assert len([item for item in submitted if item]) == 32
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(
                executor.map(
                    lambda repository: repository.consume_steering(identifier),
                    (first, second),
                )
            )
        assert sorted(len(result) for result in results) == [0, 32]
        assert len(first.messages(identifier)) == 32
        assert second.pending_steering(identifier) == []
    finally:
        first.close()
        second.close()
