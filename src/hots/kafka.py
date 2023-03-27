import json
import socket
from confluent_kafka import Producer, Consumer
import time
from . import init as it
from .instance import Instance

def acked(err, msg):
    if err is not None:
        print("Failed to deliver message: %s: %s" % (str(msg.value()), str(err)))
    else:
        print("Message produced: %s" % (str(msg.value())))


def msg_process(msg):
    # Print the current time and the message.
    time_start = time.strftime("%Y-%m-%d %H:%M:%S")
    val = msg.value()
    dval = json.loads(val)
    # print(time_start, dval)
    return(time_start, dval)


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
    Publish(it.Kafka_Producer, z, topic) # Push to Kafka


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
            'default.topic.config': {'auto.offset.reset': 'smallest'},
            'group.id': group}
    consumer = Consumer(conf)
    return consumer

def Publish(producer, msg, topic):
    jresult = json.dumps(msg)
    producer.produce(topic, key="mock_node", value=jresult, callback=acked)
    producer.flush()
