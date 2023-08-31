"""Describe needed resources for Kafka streaming platform."""

import json
import socket
import sys
import time

from confluent_kafka import Consumer, KafkaException, Producer
from confluent_kafka.admin import AdminClient
from confluent_kafka.serialization import MessageField, SerializationContext

from . import init as it
from .instance import Instance


def acked(err, msg):
    """_summary_

    :param err: _description_
    :type err: _type_
    :param msg: _description_
    :type msg: _type_
    """
    if err is not None:
        print('Failed to deliver message: %s: %s' % (str(msg.value()), str(err)))
    else:
        print('Message produced: %s' % (str(msg.value())))


def msg_process(msg, avro_deserializer):
    """_summary_

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
        val = avro_deserializer(msg.value(), SerializationContext(msg.topic(), MessageField.VALUE))
    # print(time_start, val)
    return (time_start, val)


def produce_data(my_instance: Instance, timestamp, history):
    """_summary_

    :param my_instance: _description_
    :type my_instance: Instance
    :param timestamp: _description_
    :type timestamp: _type_
    :param history: _description_
    :type history: _type_
    """
    z = {}
    if history:
        df_container = my_instance.df_indiv[
            my_instance.df_indiv.timestamp == timestamp].copy()
    else:
        df_container = my_instance.df_indiv[
            my_instance.df_indiv.timestamp == (timestamp - 1)].copy()
        df_container.loc[:, 'timestamp'] = timestamp

    df_con_dict = df_container.to_dict('records')
    z = df_con_dict
    topic = it.Kafka_topics['mock_topic']
    publish(it.Kafka_Producer, z, topic)  # Push to Kafka


def get_producer(config):
    """_summary_

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
    """_summary_

    :param config: _description_
    :type config: _type_
    :return: _description_
    :rtype: _type_
    """
    server1 = config['kafkaConf']['Consumer']['brokers'][0]
    group = config['kafkaConf']['Consumer']['group']
    conf = {'bootstrap.servers': server1,
            'max.poll.interval.ms': 1200000,
            'default.topic.config': {'auto.offset.reset': 'earliest'},
            'group.id': group}
    consumer = Consumer(conf)
    return consumer


def publish(producer, msg, topic):
    """_summary_

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
    """_summary_

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
