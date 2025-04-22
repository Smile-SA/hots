"""Describe needed resources for reading data."""

import csv
import json
import queue
import socket
import sys
import time
from pathlib import Path

import pandas as pd

try:
    from confluent_kafka import (
        Consumer, KafkaError, KafkaException, Producer, TopicPartition)
    from confluent_kafka.admin import AdminClient
    from confluent_kafka.schema_registry import SchemaRegistryClient
    from confluent_kafka.schema_registry.avro import (
        AvroDeserializer, AvroSerializer)
    from confluent_kafka.schema_registry.error import SchemaRegistryError
    from confluent_kafka.serialization import (
        MessageField, SerializationContext, StringSerializer)

    import requests
except ImportError as e:
    print(f"ImportError: {e}")
    _has_kafka = False
else:
    _has_kafka = True

from . import init as it
from . import node
from .instance import Instance


def init_reader(path):
    """Initialize the reader for data, with or without Kafka.

    :param path: initial folder path
    :type path: str
    """
    p_data = Path(path)
    if it.use_kafka:
        if not _has_kafka:
            raise ImportError('Kafka is required to do it.')
        else:
            use_schema = False
            it.avro_deserializer = connect_schema(
                use_schema, it.kafka_schema_url)
            it.csv_reader = None
            it.kafka_consumer.subscribe([it.kafka_topics['docker_topic']], on_assign=on_assign)
    else:
        it.csv_file = open(p_data / 'container_usage.csv', 'r')
        it.csv_reader = csv.reader(it.csv_file)
        it.csv_queue = queue.Queue()
        it.avro_deserializer = None
        header = next(it.csv_reader, None)
        print('Headers : ', header)
        # TODO and then ?


def on_assign(consumer, partitions):
    print("Partitions assigned:", partitions)


# TODO progress time no loop here
def get_next_data(
    current_time, tick
):
    """Get next data (one timestamp ?).

    :param current_time: Current timestamp
    :type current_time: int
    :param tick: How many new datapoints to get
    :type tick: int
    :return: Dataframe with new data
    :rtype: pd.DataFrame
    """
    print('We are in time %d and waiting for %d new datapoints ...' % (
        current_time, tick
    ))
    new_df_container = pd.DataFrame()

    it.end = False
    while not it.end:
        if it.use_kafka:

            # it.kafka_consumer.subscribe([it.kafka_topics['docker_topic']], on_assign=on_assign)
            dval = process_kafka_msg(it.avro_deserializer)
            if dval is None:
                continue
            else:
                key = list(dval.values())[0]
                value = list(dval.values())[1]
                file = list(dval.values())[2]
                new_df_container = pd.concat([
                    new_df_container, node.reassign_node(value)],
                    ignore_index=True)
                if int(key) >= current_time + tick:
                    it.end = True
            if file:
                it.s_entry = False
                break
        else:
            if it.csv_queue.empty():
                row = next(it.csv_reader, None)
                if row is None:
                    it.s_entry = False
                    break
                it.csv_queue.put(row)
            if int(it.csv_queue.queue[0][0]) <= current_time + tick:
                row = it.csv_queue.get()
                new_df_container = pd.concat([
                    new_df_container,
                    pd.DataFrame.from_records([{
                        it.tick_field: int(row[0]),
                        it.indiv_field: row[1],
                        it.host_field: row[2],
                        it.metrics[0]: float(row[3])
                    }])], ignore_index=True
                )
            else:
                new_df_container.reset_index(drop=True, inplace=True)
                it.end = True
    return new_df_container


def close_reader():
    """Close the CSV reader or Kafka consumer."""
    if it.use_kafka:
        print('Closing the stream and Kafka consumer')
        Instance.stop_stream()
        it.kafka_consumer.close()
    else:
        print('Closing the CSV file')
        it.csv_file.close()


def acked(err, msg):
    """Use Callback function for message delivery reports.

    :param err: Error object if the message delivery failed, otherwise None
    :type err: confluent_kafka.KafkaError or None
    :param msg: The message that was delivered or failed
    :type msg: confluent_kafka.Message
    """
    if err is not None:
        print('Failed to deliver message: %s: %s' %
              (str(msg.value()), str(err)))
    else:
        print('Message produced: %s' % (str(msg.value())))


def msg_process(msg, avro_deserializer):
    """Process a Kafka message and deserialize its value.

    :param msg: Kafka message to process
    :type msg: confluent_kafka.Message
    :param avro_deserializer: Function to deserialize Avro-encoded messages
    :type avro_deserializer: callable or None
    :return: A tuple containing the timestamp and the deserialized message value
    :rtype: tuple[str, Any]
    """
    # Print the current time and the message.
    time_start = time.strftime('%Y-%m-%d %H:%M:%S')
    if avro_deserializer is None:
        dval = msg.value()
        val = json.loads(dval)
    else:
        val = avro_deserializer(
            msg.value(),
            SerializationContext(msg.topic(), MessageField.VALUE))
    return (time_start, val)


def produce_data(timestamp, history):
    """Produce data to Kafka based on the given timestamp and history flag.

    :param timestamp: The current timestamp for the data
    :type timestamp: int
    :param history: Flag indicating whether to use historical data
    :type history: bool
    """
    z = {}
    if history:
        df_container = it.my_instance.df_indiv[
            it.my_instance.df_indiv.timestamp == timestamp].copy()
    else:
        df_container = it.my_instance.df_indiv[
            it.my_instance.df_indiv.timestamp == (timestamp - 1)].copy()
        df_container.loc[:, 'timestamp'] = timestamp

    df_con_dict = df_container.to_dict('records')
    z = df_con_dict
    topic = it.kafka_topics['mock_topic']
    # Push to Kafka
    publish(it.kafka_producer, z, topic)


def get_producer(config):
    """Create and return a Kafka producer.

    :param config: Configuration dictionary containing Kafka producer settings
    :type config: dict
    :return: Configured Kafka producer
    :rtype: confluent_kafka.Producer
    """
    server1 = config['kafkaConf']['Producer']['brokers'][0]
    conf = {'bootstrap.servers': server1,
            'client.id': socket.gethostname()}
    producer = Producer(conf)
    return producer


def get_consumer(config):
    """Create and return a Kafka consumer.

    :param config: Configuration dictionary containing Kafka consumer settings
    :type config: dict
    :return: Configured Kafka consumer
    :rtype: confluent_kafka.Consumer
    """
    server1 = config['kafkaConf']['Consumer']['brokers'][0]
    print(server1)
    group = config['kafkaConf']['Consumer']['group']
    conf = {'bootstrap.servers': server1,
            'max.poll.interval.ms': 300000,
            'enable.auto.commit': True,
            'auto.offset.reset': 'earliest',
            'group.id': group}
    consumer = Consumer(conf)
    return consumer


def publish(producer, msg, topic):
    """Publish a message to a Kafka topic.

    :param producer: Kafka producer instance
    :type producer: confluent_kafka.Producer
    :param msg: Message to publish
    :type msg: dict
    :param topic: Kafka topic to publish the message to
    :type topic: str
    """
    jresult = json.dumps(msg)
    producer.produce(topic, key='mock_node', value=jresult, callback=acked)
    producer.flush()


def kafka_availability(config):
    """Check the availability of the Kafka cluster.

    :param config: Configuration dictionary containing Kafka settings
    :type config: dict
    """
    server1 = config['kafkaConf']['Consumer']['brokers'][0]
    admin_client = AdminClient({'bootstrap.servers': server1})
    try:
        topic_metadata = admin_client.list_topics(timeout=10)
        if 'DockerPlacer' in topic_metadata.topics:
            print('Kafka cluster is available')
        else:
            print('Kafka cluster is not available')
    except KafkaException as e:
        print(f'Error connecting to Kafka cluster: {e}')
        sys.exit()


# Producing data stream part
def publish_stream(producer, msg, topic, avro_serializer, file_complete):
    """Publish a stream of messages to a Kafka topic.

    :param producer: Kafka producer instance
    :type producer: confluent_kafka.Producer
    :param msg: Message to publish, typically a dictionary
    :type msg: dict
    :param topic: Kafka topic to publish the message to
    :type topic: str
    :param avro_serializer: Serializer for Avro-encoded messages, or None
    :type avro_serializer: callable or None
    :param file_complete: Flag indicating whether the file is completely processed
    :type file_complete: bool
    """
    try:

        key = 'testing hots'
        timestamp, value = list(msg.items())[0]
        message = {
            'timestamp': timestamp,
            'containers': value['containers'],
            'file': True if file_complete else False
        }
        if avro_serializer is None:
            jresult = json.dumps(message)
            producer.produce(
                topic=topic, key=key, value=jresult,
                on_delivery=delivery_report)
        else:
            string_serializer = StringSerializer('utf_8')
            producer.produce(
                topic=topic, key=string_serializer(key),
                value=avro_serializer(
                    message,
                    SerializationContext(topic, MessageField.VALUE)),
                on_delivery=delivery_report)
        producer.flush()
    except KafkaException as e:
        print('Could no publish message: {}'.format(e))


def delivery_report(err, msg):
    """Use Callback function for delivery reports of Kafka messages.

    :param err: Error object if the message delivery failed, otherwise None
    :type err: confluent_kafka.KafkaError or None
    :param msg: The message that was delivered or failed
    :type msg: confluent_kafka.Message
    """
    if err is not None:
        print('Delivery failed for User record {}: {}'.format(msg.key(), err))
        return
    print('User record {} successfully produced to {} [{}] at offset {}'.
          format(msg.key(), msg.topic(), msg.partition(), msg.offset()))
    global last_offset
    last_offset = msg.offset()


def balance_offset(consumer, tp):
    """Ensure the consumer has processed all messages up to the last offset.

    :param consumer: Kafka consumer instance
    :type consumer: confluent_kafka.Consumer
    :param tp: TopicPartition object representing the topic and partition
    :type tp: confluent_kafka.TopicPartition
    """
    while True:
        committed = consumer.committed([tp])
        print('Last offset: {}, committed offset: {}.'.format(
            last_offset, committed[0].offset))
        if (last_offset - (committed[0].offset - 1)) < 10:
            # The consumer has processed all messages up to this offset
            break

        # Sleep for a short period before checking again
        time.sleep(5.0)


def connect_schema_registry(schema_str, producer_topic):
    """Connect to the schema registry and retrieve the schema.

    :param schema_str: Schema string to use for serialization
    :type schema_str: str
    :param producer_topic: Kafka topic for which the schema is retrieved
    :type producer_topic: str
    :return: AvroSerializer instance for the schema
    :rtype: confluent_kafka.schema_registry.avro.AvroSerializer
    """
    schema_registry_conf = {'url': 'http://localhost:8081'}

    schema_registry_url = schema_registry_conf['url']
    try:
        response = requests.get(schema_registry_url)

        if response.status_code == 200:
            print('Schema registry is connected.')
        else:
            print('Schema registry is not connected.')
    except Exception:
        print('Check if Schema Registry is running or disable \
              by setting use_schema to False ')
        sys.exit()

    schema_registry_client = SchemaRegistryClient(schema_registry_conf)

    try:
        schema_str = schema_registry_client.get_latest_version(
            producer_topic + '-value').schema.schema_str
    except SchemaRegistryError as e:
        # Handle schema registry error
        print(f'Error registering schema: {e}')

    return AvroSerializer(schema_registry_client, schema_str)


def csv_to_stream(data, config, use_schema=False):
    """Start the streaming for Kafka.

    :param data: Filesystem path to the input files
    :type data: str
    :param config: Configuration dict from config file
    :type config: Dict
    :param use_schema: Schema to be used by Kafka
    :type use_schema: bool
    """
    p_data = Path(data)
    rdr = csv.reader(open(p_data / 'container_usage.csv'))

    end = True
    curr_time = 0
    first_time = True
    first_line = True
    z = {}

    producer_topic = it.kafka_topics['docker_topic']

    kafka_availability(config)

    avro_serializer = None
    if use_schema:  # If you want to use avro schema (More CPU)
        avro_serializer = connect_schema_registry(
            config['kafkaConf']['schema'], producer_topic)

    producer = get_producer(config)

    consumer = get_consumer(config)
    print('producer and consumer configured')

    tp = TopicPartition('DockerPlacer', 0)
    try:
        committed = consumer.committed([tp])
        print('Partition offset before: ', committed[0].offset)
    except Exception:
        # Handle schema registry error
        print('Error connecting to consumer: ')

    while end:
        row = next(rdr, None)
        if first_line:
            first_line = False
            continue

        if row:
            curr_time = row[0]
            if first_time:
                first_time = False
                z[curr_time] = {
                    'containers': [{
                        'timestamp': row[0],
                        'container_id': row[1],
                        'machine_id': row[2],
                        'cpu': float(row[3]),
                        'mem': float(row[4]) if len(row) > 4 else None
                    }]

                }
            else:
                if curr_time in z:
                    y = {
                        'timestamp': row[0],
                        'container_id': row[1],
                        'machine_id': row[2],
                        'cpu': float(row[3]),
                        'mem': float(row[4]) if len(row) > 4 else None
                    }
                    z[curr_time]['containers'].append(y)
                else:
                    publish_stream(
                        producer=producer, msg=z, topic=producer_topic,
                        avro_serializer=avro_serializer, file_complete=False)
                    print()
                    # check wether data consumed.
                    z = {}
                    z[curr_time] = {
                        'containers': [{
                            'timestamp': row[0],
                            'container_id': row[1],
                            'machine_id': row[2],
                            'cpu': float(row[3]),
                            'mem': float(row[4]) if len(row) > 4 else None
                        }]
                    }
        else:
            if z:
                publish_stream(
                    producer=producer, msg=z, topic=producer_topic,
                    avro_serializer=avro_serializer, file_complete=True)
                # print('Another one')
            # File is over

            # check if the producer offset balanced with consumer offset
            # balance_offset(consumer, tp)
            end = False


def connect_schema(use_schema, schema_url):
    """Connect to the schema registry and retrieve the deserializer.

    :param use_schema: Flag indicating whether to use a schema registry
    :type use_schema: bool
    :param schema_url: URL of the schema registry
    :type schema_url: str
    :return: AvroDeserializer instance or None if schema is not used
    :rtype: confluent_kafka.schema_registry.avro.AvroDeserializer or None
    """
    if use_schema:
        schema_registry_client_conf = {'url': schema_url}
        schema_registry_client = SchemaRegistryClient(
            schema_registry_client_conf)

        try:
            it.kafka_schema = schema_registry_client.get_latest_version(
                it.kafka_topics['docker_topic'] + '-value'
            ).schema.schema_str
        except SchemaRegistryError as e:
            # Handle schema registry error
            print(f'Error registering schema: {e}')

        avro_deserializer = AvroDeserializer(
            schema_registry_client, it.kafka_schema)
    else:
        avro_deserializer = None
    return avro_deserializer


def process_kafka_msg(avro_deserializer):
    """Process a Kafka message and deserialize its value.

    :param avro_deserializer: Function to deserialize Avro-encoded messages
    :type avro_deserializer: callable or None
    :raises KafkaException: If there is an error in the Kafka message
    :return: Deserialized message value or None if no message is available
    :rtype: Any or None
    """
    print("Polling for message...")
    msg = it.kafka_consumer.poll(timeout=1.0)
    print(f"[CONSUMER] Assigned to: {it.kafka_consumer.assignment()}")

    if msg is None:
        return None

    if msg.error():
        if msg.error().code() == KafkaError._PARTITION_EOF:
            # End of partition event
            sys.stderr.write('%% %s [%d] reached end at offset %d\n' % (
                msg.topic(), msg.partition(), msg.offset()))
        elif msg.error().code() == KafkaError.UNKNOWN_TOPIC_OR_PART:
            sys.stderr.write(
                'Topic unknown, creating %s topic\n' % (
                    it.kafka_topics['docker_topic']))
        elif msg.error():
            print('Error in Kafka message')
            raise KafkaException(msg.error())

    else:
        print(f"✅ Message received at offset {msg.offset()}")
        (_, dval) = msg_process(msg, avro_deserializer)
        try:
            it.kafka_consumer.commit(msg, asynchronous=False)
            print(f"✅ Offset {msg.offset()} committed successfully")
        except Exception as e:
            print(f"❌ Commit failed: {e}")
        return dval


def consume_all_data(config):
    """Consume all data in the Kafka queue.

    :param config: Configuration dictionary containing Kafka settings
    :type config: dict
    """
    consumer_conf = {
        'bootstrap.servers': config['kafkaConf']['Consumer']['brokers'][0],
        'group.id': config['kafkaConf']['Consumer']['group']
    }

    consumer = Consumer(consumer_conf)

    try:
        while it.s_entry:
            consumer.subscribe([it.kafka_topics['docker_topic']])

            msg = consumer.poll(timeout=5.0)
            if msg is None:
                print('Looks like there is no data left.')
                break

            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    # End of partition event
                    sys.stderr.write(
                        '%% %s [%d] reached end at offset %d\n' %
                        (msg.topic(), msg.partition(), msg.offset())
                    )
                elif msg.error().code() == KafkaError.UNKNOWN_TOPIC_OR_PART:
                    sys.stderr.write(
                        'Topic unknown, creating %s topic\n' %
                        (it.kafka_topics['docker_topic'])
                    )
                elif msg.error():
                    raise KafkaException(msg.error())
            else:
                print(msg.value())
                msg_process(msg, None)

    except KeyboardInterrupt:
        pass
    finally:
        consumer.close()


def delete_kafka_topic(config):
    """Delete all topics in Kafka.

    :param config: Configuration dictionary containing Kafka settings
    :type config: dict
    """
    admin_client = AdminClient({
        'bootstrap.servers': config['kafkaConf']['Producer']['brokers'][0]
    })
    print(it.kafka_topics.values())
    admin_client.delete_topics(topics=list(it.kafka_topics.values()))
