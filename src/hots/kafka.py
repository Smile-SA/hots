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
    if err is not None:
        print('Failed to deliver message: %s: %s' % (str(msg.value()), str(err)))
    else:
        print('Message produced: %s' % (str(msg.value())))


def msg_process(msg, avro_deserializer):
    # Print the current time and the message.
    time_start = time.strftime('%Y-%m-%d %H:%M:%S')
    if avro_deserializer == None:
        dval = msg.value()
        val = json.loads(dval)
    else:
        val = avro_deserializer(msg.value(), SerializationContext(msg.topic(), MessageField.VALUE))
    # print(time_start, val)
    return (time_start, val)


def produce_data(my_instance: Instance, timestamp, history):
    z = {}
    if history:
        df_container = my_instance.df_indiv[my_instance.df_indiv.timestamp == timestamp].copy()
    else:
        df_container = my_instance.df_indiv[my_instance.df_indiv.timestamp == (timestamp - 1)].copy()
        df_container.loc[:, 'timestamp'] = timestamp

    df_con_dict = df_container.to_dict('records')
    z = df_con_dict
    topic = it.Kafka_topics['mock_topic']
    Publish(it.Kafka_Producer, z, topic)  # Push to Kafka


def GetProducer(config):
    server1 = config['kafkaConf']['Producer']['brokers'][0]
    conf = {'bootstrap.servers': server1,
            'client.id': socket.gethostname()}
    producer = Producer(conf)
    return producer

def GetConsumer(config):
    server1 = config['kafkaConf']['Consumer']['brokers'][0]
    group = config['kafkaConf']['Consumer']['group']
    conf = {'bootstrap.servers': server1,
            'max.poll.interval.ms': 1200000,
            'default.topic.config': {'auto.offset.reset': 'earliest'},
            'group.id': group}
    consumer = Consumer(conf)
    return consumer

def Publish(producer, msg, topic):
    jresult = json.dumps(msg)
    producer.produce(topic, key='mock_node', value=jresult, callback=acked)
    producer.flush()

def Kafka_availability(config):
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
