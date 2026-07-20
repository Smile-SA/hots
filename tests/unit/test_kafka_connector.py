"""Unit tests for KafkaConnector polling, using a scripted fake consumer.

These pin the polling/stop semantics so the flattening and stop conditions can
be refactored safely without a live broker.
"""

import json

import pytest

confluent_kafka = pytest.importorskip("confluent_kafka")

from confluent_kafka import KafkaError, KafkaException  # noqa: E402

from hots.plugins.connector.kafka_connector import KafkaConnector  # noqa: E402

TICK, INDIV, HOST, METRIC = "timestamp", "container_id", "machine_id", "cpu"


class _Err:
    def __init__(self, code):
        self._code = code

    def code(self):
        return self._code


class _Msg:
    """Fake Kafka message carrying a connector payload."""

    def __init__(self, containers=None, err=None):
        self._payload = json.dumps({"containers": containers or []}).encode("utf-8")
        self._err = err

    def value(self):
        return self._payload

    def error(self):
        return self._err


class _Consumer:
    """Returns scripted poll results, then None forever."""

    def __init__(self, script):
        self._script = list(script)
        self.polls = 0

    def poll(self, _timeout):
        self.polls += 1
        if self._script:
            return self._script.pop(0)
        return None


def _container(cid, tick, host="n_1", cpu=1.0):
    payload = {"container_id": cid, "timestamp": tick, "machine_id": host}
    if cpu is not None:
        payload[METRIC] = cpu
    return payload


def _connector(script, *, batch_size=10, timeout=5):
    c = KafkaConnector.__new__(KafkaConnector)
    c.consumer = _Consumer(script)
    c.poll_timeout = 0.0
    c.bootstrap_timeout_seconds = timeout
    c.metrics = [METRIC]
    c.tick_field, c.indiv_field, c.host_field = TICK, INDIV, HOST
    c.batch_size = batch_size
    return c


def test_flattens_payload_into_rows():
    conn = _connector([_Msg([_container("c_1", 0), _container("c_2", 0)])], batch_size=2)
    rows = conn._poll_records()

    assert rows == [
        {TICK: 0, INDIV: "c_1", HOST: "n_1", METRIC: 1.0},
        {TICK: 0, INDIV: "c_2", HOST: "n_1", METRIC: 1.0},
    ]


def test_containers_missing_the_metric_are_skipped():
    conn = _connector(
        [_Msg([_container("c_1", 0, cpu=None), _container("c_2", 0)])],
        batch_size=1,
    )
    rows = conn._poll_records()

    assert [r[INDIV] for r in rows] == ["c_2"]


def test_stops_at_batch_size():
    script = [_Msg([_container(f"c_{i}", i)]) for i in range(5)]
    conn = _connector(script, batch_size=3)
    rows = conn._poll_records()

    assert len(rows) == 3


def test_max_messages_takes_precedence():
    script = [_Msg([_container(f"c_{i}", i)]) for i in range(5)]
    conn = _connector(script, batch_size=100)
    rows = conn._poll_records(max_messages=2)

    assert len(rows) == 2


def test_stop_at_tick_keeps_polling_through_empty_polls():
    """A None poll must not end a bootstrap that is still waiting for its tick."""
    script = [
        _Msg([_container("c_1", 0)]),
        None,  # transient empty poll
        _Msg([_container("c_1", 1)]),
        _Msg([_container("c_1", 2)]),
    ]
    conn = _connector(script, batch_size=1)
    rows = conn._poll_records(stop_at_tick=2)

    assert [r[TICK] for r in rows] == [0, 1, 2]


def test_stop_at_tick_ignores_batch_size():
    """batch_size must not cut a bootstrap short."""
    script = [_Msg([_container(f"c_{i}", i)]) for i in range(4)]
    conn = _connector(script, batch_size=1)
    rows = conn._poll_records(stop_at_tick=3)

    assert [r[TICK] for r in rows] == [0, 1, 2, 3]


def test_empty_poll_ends_a_non_bootstrap_read():
    conn = _connector([_Msg([_container("c_1", 0)]), None], batch_size=100)
    rows = conn._poll_records()

    assert len(rows) == 1


def test_partition_eof_ends_a_non_bootstrap_read():
    eof = _Msg(err=_Err(KafkaError._PARTITION_EOF))
    conn = _connector([_Msg([_container("c_1", 0)]), eof], batch_size=100)
    rows = conn._poll_records()

    assert len(rows) == 1


def test_partition_eof_does_not_end_a_bootstrap():
    eof = _Msg(err=_Err(KafkaError._PARTITION_EOF))
    script = [_Msg([_container("c_1", 0)]), eof, _Msg([_container("c_1", 5)])]
    conn = _connector(script, batch_size=1)
    rows = conn._poll_records(stop_at_tick=5)

    assert [r[TICK] for r in rows] == [0, 5]


def test_real_error_is_raised():
    bad = _Msg(err=_Err(KafkaError.BROKER_NOT_AVAILABLE))
    conn = _connector([bad])

    with pytest.raises(KafkaException):
        conn._poll_records()


def test_bootstrap_gives_up_at_the_deadline():
    """Waiting for a tick that never arrives must return rather than hang."""
    import time

    # Consumer never yields data, so only the deadline can end the loop.
    conn = _connector([], batch_size=100, timeout=0.2)

    started = time.time()
    rows = conn._poll_records(stop_at_tick=99)
    elapsed = time.time() - started

    assert rows == []
    assert elapsed < 5.0, "bootstrap must honour bootstrap_timeout_seconds"
    assert conn.consumer.polls > 0
