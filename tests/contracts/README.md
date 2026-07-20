# Contract Tests

These pin the **message contract between HOTS and `hots-connector`**, so a change
to either side's message handling fails here rather than silently at integration
time.

## How it works

`fixtures/` holds the shared, transport-agnostic examples:

| File | Direction | Topic |
| --- | --- | --- |
| `telemetry_batch.json` | `hots-connector` → HOTS | `connector.parameters.topic_data` |
| `moves_batch.json` | HOTS → `hots-connector` | `connector.parameters.topic_moves` |

**These two files are meant to be byte-identical in both repositories.** Copy
them into `hots-connector` and validate from that side too: the connector should
be able to *produce* `telemetry_batch.json`'s messages and *consume*
`moves_batch.json`'s. Whichever repo changes a message shape has to change the
fixture, which breaks the other repo's suite on the next sync — that is the
point.

`test_message_contract.py` is the HOTS-side (provider) verification:

- the fixtures are self-consistent and carry the documented keys;
- `KafkaConnector` parses `telemetry_batch.json` into exactly `expected_rows`;
- a container missing the configured metric is skipped, not defaulted to zero;
- `KafkaConnector.apply_moves` publishes exactly `moves_batch.json`'s messages;
- `FileConnector` honours the *same* move contract — it both records the move
  and reflects it in `host_at()`.

Only the three Kafka assertions require the optional `kafka` extra
(`uv sync --extra dev --extra kafka`); on a plain dev install they skip
individually while the fixture-shape checks and the `FileConnector`
verification still run.

## Field names vs payload keys

The *column* names HOTS uses internally are configurable via the top-level
`data` config section. The *payload* keys in these messages
(`timestamp`, `container_id`, `machine_id`, `containers`) are fixed by this
contract and are not affected by that configuration. The metric key must match
the single entry in `data.metrics`.

## One move shape

Every producer of moves goes through `placement._move()`, which emits
`{container_name, old_host, new_host}` — the shape both connectors consume and
the one pinned by `moves_batch.json`. `test_allocation_heuristics_emit_the_wire_shape`
asserts this for the `allocation_*` heuristics, and
`test_allocation_heuristic_moves_are_understood_by_the_sink` checks a real
heuristic's output survives the sink (written *and* registered, so `host_at()`
reports the new host).

Historically the `allocation_*` heuristics emitted `{container, src, dst}`
instead, which no connector understood: such moves were written to the output
file but never registered. If you add a heuristic, build its moves with `_move()`
rather than a dict literal and that class of bug cannot come back.
