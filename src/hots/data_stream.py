"""Provide needed script to consume data for streaming platform."""

import csv
import json
import socket
import sys
import time

from confluent_kafka import Consumer, KafkaException, Producer, TopicPartition
from confluent_kafka.admin import AdminClient
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroSerializer
from confluent_kafka.schema_registry.error import SchemaRegistryError
from confluent_kafka.serialization import (MessageField, SerializationContext,
                                           StringSerializer)

import requests

last_offset = 0
UseSchema = False


def get_producer():
    """_summary_

    :return: _description_
    :rtype: _type_
    """
    server1 = '10.3.73.151:9092'

    producer_conf = {
        'bootstrap.servers': server1,
        # 'security.protocol': 'SSL',
        # 'ssl.cipher.suites': 'ECDHE-RSA-AES256-GCM-SHA384',
        # 'ssl.ca.location': '/home/muena/ssl/truststore.pem',
        # 'ssl.key.password': 'clientpass',
        # 'ssl.truststore.password': 'clientpass',
        # 'ssl.endpoint.identification.algorithm': 'None',
        # 'enable.ssl.certificate.verification': 'false',
        # 'ssl.keystore.location': '/home/muena/ssl/kafka.server.keystore.jks',
        # 'ssl.keystore.password': 'serversecret',
        'client.id': socket.gethostname(),
        'compression.type': 'gzip',  # compression to reduce the memory but more CPU
    }
    producer = Producer(producer_conf)
    return producer


def get_consumer():
    """_summary_

    :return: _description_
    :rtype: _type_
    """
    server1 = '10.3.73.151:9092'
    group = 'DockerPlacerConsumer'

    conf = {'bootstrap.servers': server1,
            'default.topic.config': {'auto.offset.reset': 'earliest'},
            # 'security.protocol':'SSL',
            # 'ssl.ca.location':'/home/muena/ssl/truststore.pem',
            # 'ssl.key.password': 'clientpass',
            # 'ssl.truststore.password': 'clientpass',
            # 'ssl.endpoint.identification.algorithm': 'None',
            # 'enable.ssl.certificate.verification': 'false',
            # 'ssl.keystore.location':'/home/muena/ssl/kafka.server.keystore.jks',
            # 'ssl.keystore.password':'serversecret',
            # 'ssl.cipher.suites':'ECDHE-RSA-AES256-GCM-SHA384',
            'group.id': group}
    consumer = Consumer(conf)
    return consumer


def publish(producer, msg, topic, avro_serializer, file_complete):
    """_summary_

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

        key = 'thesis_ex_10'
        timestamp, value = list(msg.items())[0]
        message = {
            'timestamp': timestamp,
            'containers': value['containers'],
            'file': True if file_complete else False
        }
        if avro_serializer is None:
            jresult = json.dumps(message)
            producer.produce(topic=topic, key=key, value=jresult, on_delivery=delivery_report)
        else:
            string_serializer = StringSerializer('utf_8')
            producer.produce(
                topic=topic, key=string_serializer(key),
                value=avro_serializer(message, SerializationContext(topic, MessageField.VALUE)),
                on_delivery=delivery_report)
        producer.flush()
    except KafkaException as e:
        print('Could no publish message: {}'.format(e))


def delivery_report(err, msg):
    """_summary_

    :param err: _description_
    :type err: _type_
    :param msg: _description_
    :type msg: _type_
    """
    if err is not None:
        print('Delivery failed for User record {}: {}'.format(msg.key(), err))
        return
    print('User record {} successfully produced to {} [{}] at offset {}'.format(
        msg.key(), msg.topic(), msg.partition(), msg.offset()))
    global last_offset
    last_offset = msg.offset()


def balance_offset(consumer, tp):
    """_summary_

    :param consumer: _description_
    :type consumer: _type_
    :param tp: _description_
    :type tp: _type_
    """
    while True:
        # Retrieve the latest committed offset for the topic partition being consumed
        committed = consumer.committed([tp])
        # offset is position of the next message that the consumer will read, we substract by 1
        print('Last offset: {}, committed offset: {}.'.format(last_offset, committed[0].offset))
        if (last_offset - (committed[0].offset - 1)) < 10:
            # The consumer has processed all messages up to this offset
            break

        # Sleep for a short period before checking again
        time.sleep(5.0)


def kafka_availability():
    """_summary_"""
    admin_client = AdminClient({
        'bootstrap.servers': '10.3.73.151:9092',
        #  'security.protocol':'SSL',
        #  'ssl.ca.location':'/home/muena/ssl',
    })
    try:
        topic_metadata = admin_client.list_topics(timeout=10)
        if 'DockerPlacer' in topic_metadata.topics:
            print('Kafka cluster is available')
        else:
            print('Kafka cluster is not available')
    except KafkaException as e:
        print(f'Error connecting to Kafka cluster: {e}')
        sys.exit()


def main():
    """_summary_"""
    schema_str = """
        {
        "namespace": "com.example",
        "type": "record",
        "name": "Person",
        "fields": [
            {
                "type": "string",
                "name": "timestamp"
            },
            {
            "type": {
                "type": "array",
                "items": {
                "type": "record",
                "name": "Container",
                "namespace": "com.smile.containers",
                "fields": [
                    {
                    "type": "string",
                    "name": "timestamp"
                    },
                    {
                    "type": "string",
                    "name": "container_id"
                    },
                    {
                    "type": "string",
                    "name": "machine_id"
                    },
                    {
                    "type": "float",
                    "name": "cpu"
                    }
                ]
                }
            },
            "name": "containers"
            },
            {
                "type": "boolean",
                "name": "file"
            }
        ]
        }
        """
    file_path = '/home/etilec/Documents/code/temp_hots/mock_container_usage.csv'
    rdr = csv.reader(open(file_path))

    end = True
    curr_time = 0
    first_time = True
    first_line = True
    z = {}

    kafka_conf = {
        'docker_topic': 'DockerPlacer',
    }
    producer_topic = kafka_conf['docker_topic']

    kafka_availability()  # Wait for 10 seconds to see if kafka broker is up!

    avro_serializer = None
    if UseSchema:  # If you want to use avro schema (More CPU!!)

        # schema_registry_conf = {'url': 'http://localhost:8081'}
        schema_registry_conf = {'url': 'http://10.3.73.151:8081'}

        schema_registry_url = schema_registry_conf['url']
        try:
            response = requests.get(schema_registry_url)

            if response.status_code == 200:
                print('Schema registry is connected.')
            else:
                print('Schema registry is not connected.')
        except:
            print(f'Check if Schema Registry is running or disable by setting UseSchema to False ')
            sys.exit()

        schema_registry_client = SchemaRegistryClient(schema_registry_conf)

        try:
            schema_str = schema_registry_client.get_latest_version(
                producer_topic + '-value').schema.schema_str
        except SchemaRegistryError as e:
            # Handle schema registry error
            print(f'Error registering schema: {e}')

        avro_serializer = AvroSerializer(schema_registry_client, schema_str)
    producer = get_producer()

    consumer = get_consumer()
    print('producer and consumer configured')

    tp = TopicPartition('DockerPlacer', 0)
    try:
        committed = consumer.committed([tp])
        print('Partition offset before: ', committed[0].offset)
    except:
        # Handle schema registry error
        print(f'Error connecting to consumer: ')

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
                    publish(
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
                publish(
                    producer=producer, msg=z, topic=producer_topic,
                    avro_serializer=avro_serializer, file_complete=True)
                # print('Another one')
            # File is over

            # check if the producer offset balanced with consumer offset
            # balance_offset(consumer, tp)
            end = False


if __name__ == '__main__':
    main()
