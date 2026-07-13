"""Unit tests for the file connector CSV parsing."""

import pytest

from hots.plugins.connector.file_connector import FileConnector


def _write_csv(path, header, rows):
    lines = [",".join(header)] + [",".join(str(v) for v in r) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_params(data_dir, tmp_path, **overrides):
    params = {
        "data_folder": str(data_dir),
        "file_name": "container_usage.csv",
        "tick_field": "timestamp",
        "individual_field": "container_id",
        "host_field": "machine_id",
        "metrics": ["cpu"],
        "sep_time": 1,
        "tick_increment": 1,
        "outfile": str(tmp_path / "moves.jsonl"),
    }
    params.update(overrides)
    return params


def test_reordered_and_extra_columns(tmp_path):
    """Columns are read by name: order and extra columns must not matter."""
    _write_csv(
        tmp_path / "container_usage.csv",
        ["machine_id", "cpu", "container_id", "mem", "timestamp", "extra"],
        [
            ["n_1", 1.5, "c_1", 12, 0, "x"],
            ["n_2", 2.5, "c_2", 13, 0, "y"],
            ["n_1", 3.5, "c_1", 14, 1, "z"],
        ],
    )
    _write_csv(tmp_path / "node_meta.csv", ["machine_id", "cpu"], [["n_1", 6], ["n_2", 6]])

    connector = FileConnector(_make_params(tmp_path, tmp_path))
    df_indiv, _, df_meta = connector.load_initial()
    connector.close()

    assert list(df_indiv["container_id"]) == ["c_1", "c_2", "c_1"]
    assert list(df_indiv["machine_id"]) == ["n_1", "n_2", "n_1"]
    assert list(df_indiv["timestamp"]) == [0, 0, 1]
    assert list(df_indiv["cpu"]) == [1.5, 2.5, 3.5]
    assert set(df_meta["machine_id"]) == {"n_1", "n_2"}


def test_missing_column_fails_fast(tmp_path):
    """A CSV missing a configured column must raise instead of producing garbage."""
    _write_csv(
        tmp_path / "container_usage.csv",
        ["timestamp", "container_id", "machine_id"],  # no 'cpu'
        [[0, "c_1", "n_1"]],
    )
    _write_csv(tmp_path / "node_meta.csv", ["machine_id", "cpu"], [["n_1", 6]])

    with pytest.raises(ValueError, match="cpu"):
        FileConnector(_make_params(tmp_path, tmp_path))


def test_outfile_reset_is_per_instance(tmp_path):
    """Two connectors sharing an outfile each reset it (no process-global guard)."""
    _write_csv(
        tmp_path / "container_usage.csv",
        ["timestamp", "container_id", "machine_id", "cpu"],
        [[0, "c_1", "n_1", 1.0]],
    )
    _write_csv(tmp_path / "node_meta.csv", ["machine_id", "cpu"], [["n_1", 6]])
    outfile = tmp_path / "moves.jsonl"

    a = FileConnector(_make_params(tmp_path, tmp_path, outfile=str(outfile)))
    a.apply_moves([{"container_name": "c_1", "old_host": "n_1", "new_host": "n_2"}], 0)
    a.close()
    assert outfile.read_text() != ""

    # A second connector on the same outfile must truncate it again.
    b = FileConnector(_make_params(tmp_path, tmp_path, outfile=str(outfile)))
    b.close()
    assert outfile.read_text() == ""


def test_load_next_windows(tmp_path):
    """load_next returns rows per tick window and None when data is exhausted."""
    _write_csv(
        tmp_path / "container_usage.csv",
        ["timestamp", "container_id", "machine_id", "cpu"],
        [
            [0, "c_1", "n_1", 1.0],
            [1, "c_1", "n_1", 2.0],
            [2, "c_1", "n_1", 3.0],
        ],
    )
    _write_csv(tmp_path / "node_meta.csv", ["machine_id", "cpu"], [["n_1", 6]])

    connector = FileConnector(_make_params(tmp_path, tmp_path, sep_time=0))
    df0, _, _ = connector.load_initial()
    assert list(df0["timestamp"]) == [0]

    df1 = connector.load_next()
    assert list(df1["timestamp"]) == [1]
    df2 = connector.load_next()
    assert list(df2["timestamp"]) == [2]
    assert connector.load_next() is None
    connector.close()
