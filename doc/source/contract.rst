.. _contract:

====================================
Message contract with hots-connector
====================================

With the Kafka connector, :term:`hots` exchanges JSON messages with
`hots-connector <https://github.com/Smile-SA/hots-connector>`_, which observes a live environment
and applies the moves. Both projects test against the same example messages, kept byte-identical
in both repositories under :file:`tests/contracts/fixtures/`.

Telemetry
=========

hots-connector to :term:`hots`, on :code:`connector.parameters.topic_data`. One message per
batch:

.. code-block:: json

   {"v": 1, "containers": [
     {"timestamp": 1700000000, "container_id": "c_1", "machine_id": "vm1", "cpu": 0.42}
   ]}

- :code:`containers` is the only key besides :code:`v`.
- The payload keys :code:`timestamp`, :code:`container_id` and :code:`machine_id` are fixed:
  the :code:`data` section of the configuration renames the columns inside :term:`hots`, not
  these keys.
- The metric key must be the entry of :code:`data.metrics`. A container without a value for it
  is skipped, never counted as zero.

Moves
=====

:term:`hots` to hots-connector, on :code:`connector.parameters.topic_moves`. One message per move:

.. code-block:: json

   {"v": 1, "container_name": "c_1", "old_host": "vm1", "new_host": "vm2", "time": 1700000030}

:code:`old_host` is :code:`null` for an initial placement.

Schema version
==============

Both messages carry :code:`"v": 1`. A change to either shape is made in both projects at once:
update the fixtures in both repositories, keep them byte-identical, and bump :code:`v` if the
change is incompatible. :file:`tests/contracts/README.md` describes the tests on the
:term:`hots` side.

========================  ===============
hots                      hots-connector
========================  ===============
2.1                       0.1
========================  ===============
