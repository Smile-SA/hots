import json
import sys
import time
import socket
from confluent_kafka import Consumer, KafkaError, KafkaException
from confluent_kafka import Producer

import pandas as pd
from pathlib import Path

Sentry = True

def SignalHandler_SIGINT(SignalNumber,Frame):
    global Sentry 
    Sentry = False

def msg_process(msg):
    # Print the current time and the message.
    time_start = time.strftime("%Y-%m-%d %H:%M:%S")
    val = msg.value()
    dval = json.loads(val)
    # print(time_start, dval)
    return time_start, dval

def acked(err, msg):
    if err is not None:
        print("Failed to deliver message: %s: %s" % (str(msg.value()), str(err)))
    else:
        print("Message produced: %s" % (str(msg.value())))


def GetProducer():
    conf = {'bootstrap.servers': "localhost:9092",
            'client.id': socket.gethostname()}
    producer = Producer(conf)
    return producer

def GetConsumer():
    
    conf = {'bootstrap.servers': 'localhost:9092',
            'default.topic.config': {'auto.offset.reset': 'smallest'},
            'group.id': socket.gethostname()}
    consumer = Consumer(conf)
    return consumer

def IntializeData(path):
    # Get data from the csv based on file name ?
    # extract cpu/gpu data from file using new node values  
    df_mock_indiv = pd.read_csv(Path(path) / 'container_usage.csv', index_col=False)
    df_mock_indiv.reset_index(drop=True, inplace=True)
    df_mock_indiv.set_index(['container_id'], inplace=True)
    df_mock_indiv.sort_index(inplace=True)
    # df_mock_indiv.sort_values('timestamp', inplace=True)
    # df_mock_indiv.set_index(['timestamp', 'container_id'], inplace=True, drop=False)

    return df_mock_indiv

def GetMockData(timestamp, new_data, c_info):
    mock_data = new_data['machine_id']
    
    for c in c_info:
        mock_data[c['container_id']] = c['machine_id']

    new_data['machine_id'] = mock_data

    z = {}

    first = True
    cpu_stats = new_data['cpu']
    for k, v in new_data['machine_id'].items():
        # k is the container
        # v is the machine_id
        container_id = k
        machine_id = v
        cpu = cpu_stats[k]
        if first:
            z[timestamp] = {
                                    "containers": [{
                                    "container_id": container_id,
                                    "machine_id": machine_id,
                                    "cpu": cpu
                                }]
                                
                            }
            first = False
        else:
            y  = {
                            "container_id": container_id,
                            "machine_id": machine_id,
                            "cpu": cpu
                         }
            z[timestamp]['containers'].append(y)
    return z


def main():
    
    kafkaConf = {
      "docker_topic": "DockerPlacer",
      "mock_topic": "MockPlacement"
    }
    
    consumer_topic = kafkaConf['mock_topic']
    producer_topic = kafkaConf['docker_topic']

    producer = GetProducer()

    consumer = GetConsumer()
    path = 'tests/data/generated_7'
    df = IntializeData(path)
    
    try: 
        while Sentry:

            consumer.subscribe([consumer_topic])

            msg = consumer.poll(timeout=5.0)
            if msg is None:
                continue

            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    # End of partition event
                    sys.stderr.write('%% %s [%d] reached end at offset %d\n' %
                                        (msg.topic(), msg.partition(), msg.offset()))
                elif msg.error().code() == KafkaError.UNKNOWN_TOPIC_OR_PART:
                    sys.stderr.write('Topic unknown, creating %s topic\n' %
                                        (consumer_topic))
                elif msg.error():
                    raise KafkaException(msg.error())
            else:
                _, dval= msg_process(msg)
                
                try:
                    if dval:
                        timestamp = dval[0]['timestamp']
                        print('timestamp: ', timestamp)
                        next_data = df[df['timestamp']== int(timestamp)].copy()
                        if not next_data.empty:
                            c_info = next_data.to_dict()
                            js = GetMockData(timestamp, c_info, dval)
                            
                        else: 
                            js = {}
                        jresult = json.dumps(js)
                        producer.produce(producer_topic, key="mock_node", value=jresult, callback=acked)
                        producer.flush()

                except TypeError:
                    sys.exit()

    except KeyboardInterrupt:
        pass
    finally:
        consumer.close()
            


if __name__ == "__main__":
    main()