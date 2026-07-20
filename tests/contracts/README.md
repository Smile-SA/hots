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

## Known gap: two move shapes

`PlacementPlugin.move_container` — the streaming path — emits the wire shape
`{container_name, old_host, new_host}` that connectors consume.

The `allocation_*` heuristics (`allocation_distant_pairwise`, `allocation_ffd`,
`allocation_spread`, `colocalize_clusters`, `find_substitution`) instead emit
`{container, src, dst}`. `FileConnector.add_moves` reads `container_name` /
`new_host`, so those moves are written to the output file but **never
registered**, and `host_at()` keeps returning the old host. `KafkaConnector`
would publish them verbatim, in a shape the connector does not understand.

`test_allocation_heuristic_moves_are_understood_by_the_sink` is a `strict=True`
xfail pinning this. When the shapes are unified it will start passing and fail
the suite, prompting removal of the xfail.
