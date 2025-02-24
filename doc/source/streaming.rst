.. _streaming:

==============
Streaming Data
==============

This document describes how the Kafka streaming platform is integrated into the application, detailing the steps involved in producing and consuming messages.

Overview
--------
The main function in the application is responsible for reading data from a CSV file, creating a Kafka producer and consumer, and publishing messages to a Kafka topic. The following steps outline the integration process:

1. The CSV file is read row by row.
2. Data from each row is extracted and used to create a message.
3. The message is published to Kafka using the `Publish` function.
4. The message follows an Avro schema defined in the `schema_str` variable.

Kafka Components
----------------

Producer and Consumer Initialization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **`GetProducer` Function**

  - Initializes and returns a Kafka producer object.

- **`GetConsumer` Function**

  - Initializes and returns a Kafka consumer object.

Message Publishing
~~~~~~~~~~~~~~~~~~

- **`Publish` Function**

  - Accepts a producer object, a message, a topic, and an Avro serializer.
  - Extracts data from the message and serializes it using the Avro schema.
  - Publishes the message to the specified Kafka topic.
  - Uses the message's timestamp as the key.

Delivery Reporting
~~~~~~~~~~~~~~~~~~

- **`delivery_report` Function**

  - A callback function triggered when a message is delivered to Kafka.
  - Checks whether the delivery was successful.
  - Updates the `last_offset` variable with the offset of the delivered message.

Consumer Offset Balancing
~~~~~~~~~~~~~~~~~~~~~~~~~

- **`balance_offset` Function**

  - Retrieves the latest committed offset for the partition being consumed.
  - Compares the latest offset with `last_offset`.
  - Ensures the consumer has processed all messages up to the committed offset.

This integration ensures efficient message processing using Kafka while maintaining data consistency and offset balancing for consumers.
