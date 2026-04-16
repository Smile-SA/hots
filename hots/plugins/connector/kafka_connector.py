"""Kafka connector plugin for HOTS."""

from __future__ import annotations

import json
import time

import pandas as pd
from confluent_kafka import Consumer, KafkaError, KafkaException, Producer

from hots.core.interfaces import ConnectorPlugin


class KafkaConnector(ConnectorPlugin):
    """Connector plugin that reads telemetry from Kafka and publishes moves."""

    def __init__(self, params):
        self.params = params

        self.tick_field = params["tick_field"]
        self.indiv_field = params["individual_field"]
        self.host_field = params["host_field"]
        self.metrics = params["metrics"]

        self.tick_increment = int(params.get("tick_increment", 1))
        self.sep_time = int(params.get("sep_time", 10))
        self.batch_size = int(params.get("batch_size", 1))
        self.poll_timeout = float(params.get("poll_timeout", 1.0))
        self.bootstrap_timeout_seconds = int(params.get("bootstrap_timeout_seconds", 120))

        consumer_conf = dict(params["consumer_conf"])
        self.consumer = Consumer(consumer_conf)
        self.topic_data = params["topic_data"]
        self.topic_moves = params["topic_moves"]
        self.consumer.subscribe([self.topic_data])

        self.producer = Producer({"bootstrap.servers": params["bootstrap.servers"]})

        self.current_time = 0

    def _poll_records(self, *, stop_at_tick: int | None = None, max_messages: int | None = None):
        """
        Poll Kafka and flatten connector payloads into HOTS rows.

        If stop_at_tick is provided, keep polling until we have seen records whose
        timestamp reaches that tick.
        """
        records = []
        seen_max_tick = None
        deadline = time.time() + self.bootstrap_timeout_seconds

        while True:
            if time.time() > deadline:
                break

            msg = self.consumer.poll(self.poll_timeout)
            if msg is None:
                if stop_at_tick is None:
                    break
                continue

            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    if stop_at_tick is None:
                        break
                    continue
                raise KafkaException(msg.error())

            payload = json.loads(msg.value())

            for c in payload.get("containers", []):
                metric_name = self.metrics[0]
                metric_value = c.get(metric_name)
                if metric_value is None:
                    continue

                tick_value = int(c["timestamp"])
                row = {
                    self.tick_field: tick_value,
                    self.indiv_field: c["container_id"],
                    self.host_field: c["machine_id"],
                    metric_name: metric_value,
                }
                records.append(row)

                if seen_max_tick is None or tick_value > seen_max_tick:
                    seen_max_tick = tick_value

            if max_messages is not None and len(records) >= max_messages:
                break

            if (
                stop_at_tick is not None
                and seen_max_tick is not None
                and seen_max_tick >= stop_at_tick
            ):
                break

            if stop_at_tick is None and len(records) >= self.batch_size:
                break

        return records

    def _build_df_meta(self, df_indiv: pd.DataFrame) -> pd.DataFrame:
        """
        Build node metadata for HOTS placement.

        Preferred source is connector.parameters["node_capacities"] = {"vm1": 2.0, ...}
        Fallback is to use the max observed host usage in the bootstrap window.
        """
        metric = self.metrics[0]
        capacities = self.params.get("node_capacities")

        if capacities:
            rows = [{self.host_field: host, metric: float(cap)} for host, cap in capacities.items()]
            return pd.DataFrame(rows)

        df_host = df_indiv.groupby([self.host_field, self.tick_field], as_index=False)[metric].sum()
        df_meta = (
            df_host.groupby(self.host_field, as_index=False)[metric]
            .max()
            .rename(columns={metric: metric})
        )
        return df_meta

    def load_initial(self):
        """
        Bootstrap the initial HOTS window from Kafka.

        We wait until we have consumed telemetry up to sep_time.
        """
        records = self._poll_records(stop_at_tick=self.sep_time)

        if not records:
            raise RuntimeError(
                "Kafka bootstrap failed: no telemetry received before bootstrap timeout."
            )

        df_indiv = pd.DataFrame.from_records(records).sort_values(
            by=[self.tick_field, self.indiv_field]
        )
        df_meta = self._build_df_meta(df_indiv)

        self.current_time = int(df_indiv[self.tick_field].max())
        return df_indiv, None, df_meta

    def load_next(self):
        """Poll Kafka and return the next batch of telemetry rows."""
        records = self._poll_records(max_messages=self.batch_size)
        if not records:
            return None

        df = pd.DataFrame.from_records(records).sort_values(by=[self.tick_field, self.indiv_field])
        self.current_time = int(df[self.tick_field].max())
        return df

    def apply_moves(self, moves, current_time: int | None = None):
        """Publish move dicts to the Kafka move topic."""
        if not moves:
            return

        for move in moves:
            payload = dict(move)
            if current_time is not None and "time" not in payload:
                payload["time"] = int(current_time)
            self.producer.produce(self.topic_moves, json.dumps(payload).encode("utf-8"))

        self.producer.flush()
