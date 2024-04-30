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
except ImportError:
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
    else:
        it.csv_file = open(p_data / 'container_usage.csv', 'r')
        it.csv_reader = csv.reader(it.csv_file)
        it.csv_queue = queue.Queue()
        it.avro_deserializer = None
        header = next(it.csv_reader, None)
        print('Headers : ', header)
        # TODO and then ?


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

            it.kafka_consumer.subscribe([it.kafka_topics['docker_topic']])
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
    """Summary.

    :param err: _description_
    :type err: _type_
    :param msg: _description_
    :type msg: _type_
    """
    if err is not None:
        print('Failed to deliver message: %s: %s' %
              (str(msg.value()), str(err)))
    else:
        print('Message produced: %s' % (str(msg.value())))


def msg_process(msg, avro_deserializer):
    """Summary.

    :param msg: _description_
    :type msg: _type_
    :param avro_deserializer: _description_
    :type avro_deserializer: _type_
    :return: _description_
    :rtype: _type_
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
    """Summary.

    :param timestamp: _description_
    :type timestamp: _type_
    :param history: _description_
    :type history: _type_
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
    """Summary.

    :param config: _description_
    :type config: _type_
    :return: _description_
    :rtype: _type_
    """
    server1 = config['kafkaConf']['Producer']['brokers'][0]
    conf = {'bootstrap.servers': server1,
            'client.id': socket.gethostname()}
    producer = Producer(conf)
    return producer


def get_consumer(config):
    """Summary.

    :param config: _description_
    :type config: _type_
    :return: _description_
    :rtype: _type_
    """
    server1 = config['kafkaConf']['Consumer']['brokers'][0]
    print(server1)
    group = config['kafkaConf']['Consumer']['group']
    conf = {'bootstrap.servers': server1,
            'max.poll.interval.ms': 1200000,
            'default.topic.config': {'auto.offset.reset': 'earliest'},
            'group.id': group}
    consumer = Consumer(conf)
    return consumer


def publish(producer, msg, topic):
    """Summary.

    :param producer: _description_
    :type producer: _type_
    :param msg: _description_
    :type msg: _type_
    :param topic: _description_
    :type topic: _type_
    """
    jresult = json.dumps(msg)
    producer.produce(topic, key='mock_node', value=jresult, callback=acked)
    producer.flush()


def kafka_availability(config):
    """Summary.

    :param config: _description_
    :type config: _type_
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
    """Summary.

    :param producer: _description_
    :type producer: _type_
    :param msg: _description_
    :type msg: _type_
    :param topic: _description_
    :type topic: _type_
    :param avro_serializer: _description_
    :type avro_serializer: _type_
    :param file_complete: _description_
    :type file_complete: _type_
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
    """Summary.

    :param err: _description_
    :type err: _type_
    :param msg: _description_
    :type msg: _type_
    """
    if err is not None:
        print('Delivery failed for User record {}: {}'.format(msg.key(), err))
        return
    print('User record {} successfully produced to {} [{}] at offset {}'.
          format(msg.key(), msg.topic(), msg.partition(), msg.offset()))
    global last_offset
    last_offset = msg.offset()


def balance_offset(consumer, tp):
    """Summary.

    :param consumer: _description_
    :type consumer: _type_
    :param tp: _description_
    :type tp: _type_
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
    """Summary.

    :param schema_str: _description_
    :type schema_str: _type_
    :param producer_topic: _description_
    :type producer_topic: _type_
    :return: _description_
    :rtype: _type_
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
    """Summary.

    :param use_schema: _description_
    :type use_schema: _type_
    :param use_schema_url: _description_
    :type use_schema_url: _type_
    :return: _description_
    :rtype: _type_
    """
    if use_schema:
        schema_registry_client_conf = {'url': schema_url}
        schema_registry_client = SchemaRegistryClient(
            schema_registry_client_conf)

        try:
            it.kafka_schema = schema_registry_client.get_latest_version(
                it.kafka_topics['docker_topic'] + '-value'
            ).schema.it.kafka_schema
        except SchemaRegistryError as e:
            # Handle schema registry error
            print(f'Error registering schema: {e}')

        avro_deserializer = AvroDeserializer(
            schema_registry_client, it.kafka_schema)
    else:
        avro_deserializer = None
    return avro_deserializer


def process_kafka_msg(avro_deserializer):
    """Summary.

    :param avro_deserializer: _description_
    :type avro_deserializer: _type_
    :raises KafkaException: _description_
    :return: _description_
    :rtype: _type_
    """
    msg = it.kafka_consumer.poll(timeout=1.0)

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
            print('error message here')
            raise KafkaException(msg.error())

    else:
        (_, dval) = msg_process(msg, avro_deserializer)
        return dval


def consume_all_data(config):
    """Consume all data in queue.

    :param config: _description_
    :type config: _type_
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
                print('Look like there is no data left.')
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

    :param config: _description_
    :type config: _type_
    """
    admin_client = AdminClient({
        'bootstrap.servers': [config['kafkaConf']['Producer']['brokers'][0]]
    })
    print(it.kafka_topics.values())
    admin_client.delete_topics(topics=list(it.kafka_topics.values()))
