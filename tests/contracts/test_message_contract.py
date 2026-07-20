"""Contract tests for the HOTS <-> hots-connector message formats.

These validate HOTS against the shared fixtures in ``fixtures/``. The same
files are meant to be checked into hots-connector and validated from that side,
so a change to either repo's message handling fails here rather than silently at
integration time. See README.md in this directory.
"""

import importlib.util
import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"

# Only the Kafka half needs the optional extra; the fixture-shape checks and the
# FileConnector verification must run on a plain dev install.
_HAS_KAFKA = importlib.util.find_spec("confluent_kafka") is not None
requires_kafka = pytest.mark.skipif(not _HAS_KAFKA, reason="Kafka contract needs the 'kafka' extra")


def _load(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def telemetry():
    return _load("telemetry_batch.json")


@pytest.fixture(scope="module")
def moves():
    return _load("moves_batch.json")


# --------------------------------------------------------------------------
# Fixture self-consistency (runs without confluent_kafka)
# --------------------------------------------------------------------------


def test_telemetry_fixture_shape(telemetry):
    """The inbound payload is {"containers": [...]} with fixed key names."""
    assert telemetry["direction"] == "connector_to_hots"
    assert telemetry["messages"], "fixture must carry at least one message"

    for msg in telemetry["messages"]:
        assert set(msg) == {"containers"}
        for c in msg["containers"]:
            assert {"timestamp", "container_id", "machine_id"} <= set(c)
            assert isinstance(c["timestamp"], int)
            assert isinstance(c["container_id"], str)
            assert isinstance(c["machine_id"], str)


def test_moves_fixture_shape(moves):
    """The outbound move carries container_name/old_host/new_host (+ time)."""
    assert moves["direction"] == "hots_to_connector"
    required = set(moves["required_keys"])
    assert required == {"container_name", "old_host", "new_host"}

    for msg in moves["messages"]:
        assert required <= set(msg)
        assert set(msg) <= required | {"time"}


# --------------------------------------------------------------------------
# Provider side: HOTS must parse/emit exactly these shapes
# --------------------------------------------------------------------------

TICK, INDIV, HOST, METRIC = "timestamp", "container_id", "machine_id", "cpu"


class _Msg:
    def __init__(self, payload):
        self._payload = json.dumps(payload).encode("utf-8")

    def value(self):
        return self._payload

    def error(self):
        return None


class _Consumer:
    def __init__(self, messages):
        self._queue = [_Msg(m) for m in messages]

    def poll(self, _timeout):
        return self._queue.pop(0) if self._queue else None


class _Producer:
    def __init__(self):
        self.published = []
        self.flushed = 0

    def produce(self, topic, value):
        self.published.append((topic, json.loads(value.decode("utf-8"))))

    def flush(self):
        self.flushed += 1


def _connector(messages=(), *, batch_size=100):
    from hots.plugins.connector.kafka_connector import KafkaConnector

    c = KafkaConnector.__new__(KafkaConnector)
    c.consumer = _Consumer(list(messages))
    c.producer = _Producer()
    c.poll_timeout = 0.0
    c.bootstrap_timeout_seconds = 5
    c.metrics = [METRIC]
    c.tick_field, c.indiv_field, c.host_field = TICK, INDIV, HOST
    c.batch_size = batch_size
    c.topic_moves = "hots_moves"
    return c


@requires_kafka
def test_hots_parses_the_telemetry_fixture(telemetry):
    """Provider verification: the documented payload yields the documented rows."""
    conn = _connector(telemetry["messages"])
    rows = conn._poll_records()

    assert rows == telemetry["expected_rows"]


@requires_kafka
def test_container_without_the_metric_is_skipped(telemetry):
    """Documented in the fixture notes: absent metric means skip, not zero."""
    msg = {
        "containers": [
            {"timestamp": 0, "container_id": "c_1", "machine_id": "n_1"},  # no cpu
            {"timestamp": 0, "container_id": "c_2", "machine_id": "n_1", "cpu": 1.0},
        ]
    }
    conn = _connector([msg])
    rows = conn._poll_records()

    assert [r[INDIV] for r in rows] == ["c_2"]


@requires_kafka
def test_hots_publishes_the_move_fixture_shape(moves):
    """Provider verification: apply_moves emits exactly the documented keys."""
    conn = _connector()
    tick = moves["messages"][0]["time"]

    # Publish the heuristic's output (the move without the connector's timestamp)
    payloads = [{k: m[k] for k in moves["required_keys"]} for m in moves["messages"]]
    conn.apply_moves(payloads, tick)

    published = [body for _topic, body in conn.producer.published]
    assert published == moves["messages"]
    assert {topic for topic, _ in conn.producer.published} == {"hots_moves"}
    assert conn.producer.flushed == 1


def test_file_connector_consumes_the_same_move_shape(moves, tmp_path):
    """Both sinks must agree on the move contract, not just the Kafka one."""
    from hots.plugins.connector.file_connector import FileConnector

    usage = tmp_path / "container_usage.csv"
    usage.write_text("timestamp,container_id,machine_id,cpu\n0,c_1,n_1,1.0\n", encoding="utf-8")
    (tmp_path / "node_meta.csv").write_text("machine_id,cpu\nn_1,6\n", encoding="utf-8")
    outfile = tmp_path / "moves.jsonl"

    conn = FileConnector(
        {
            "data_folder": str(tmp_path),
            "tick_field": TICK,
            "individual_field": INDIV,
            "host_field": HOST,
            "metrics": [METRIC],
            "sep_time": 0,
            "outfile": str(outfile),
        }
    )

    payloads = [{k: m[k] for k in moves["required_keys"]} for m in moves["messages"]]
    tick = moves["messages"][0]["time"]
    conn.apply_moves(payloads, tick)
    conn.close()

    written = [json.loads(line) for line in outfile.read_text().splitlines() if line.strip()]
    expected_names = [m["container_name"] for m in moves["messages"]]
    assert [w["container_name"] for w in written] == expected_names

    # The move must actually be registered, not just logged: a later read of the
    # moved container has to report its new host.
    for move in moves["messages"]:
        assert conn.host_at(move["container_name"], tick, move["old_host"]) == move["new_host"]


@pytest.mark.xfail(
    reason=(
        "KNOWN GAP: the allocation_* heuristics (used by PlacementPlugin.initial) "
        "emit {container, src, dst}, but connectors consume "
        "{container_name, old_host, new_host}. Such moves are written to the sink "
        "but never registered, so host_at() keeps reporting the old host. "
        "Remove this xfail once the two shapes are unified."
    ),
    strict=True,
)
def test_allocation_heuristic_moves_are_understood_by_the_sink(tmp_path):
    """The shape produced by allocation_* should satisfy the same move contract."""
    from hots.plugins.connector.file_connector import FileConnector

    (tmp_path / "container_usage.csv").write_text(
        "timestamp,container_id,machine_id,cpu\n0,c_1,n_1,1.0\n", encoding="utf-8"
    )
    (tmp_path / "node_meta.csv").write_text("machine_id,cpu\nn_1,6\n", encoding="utf-8")
    outfile = tmp_path / "moves.jsonl"

    conn = FileConnector(
        {
            "data_folder": str(tmp_path),
            "tick_field": TICK,
            "individual_field": INDIV,
            "host_field": HOST,
            "metrics": [METRIC],
            "sep_time": 0,
            "outfile": str(outfile),
        }
    )

    # This is exactly what allocation_distant_pairwise/ffd/spread return.
    conn.apply_moves([{"container": "c_1", "src": "n_1", "dst": "n_2"}], 0)
    conn.close()

    assert conn.host_at("c_1", 0, "n_1") == "n_2"
