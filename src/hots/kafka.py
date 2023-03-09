import json
import socket
from confluent_kafka import Producer, Consumer
import time
import init as it
import instance as inst


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


def produce_data(my_instance: inst.Instance, timestamp, history):

    if history:
        df_container = my_instance.df_indiv[my_instance.df_indiv.timestamp == timestamp].copy()
    else:
        df_container = my_instance.df_mock_indiv[my_instance.df_mock_indiv.timestamp == timestamp].copy()
        df_old = my_instance.df_indiv[my_instance.df_indiv.timestamp == (timestamp - 1)].copy()
        df_old.reset_index(drop=True, inplace=True)
        df_old.loc[:, 'timestamp'] = timestamp
        df_old.set_index([it.tick_field, it.indiv_field], inplace=True, drop=False)
        df_container.sort_index(inplace=True)
        df_old.sort_index(inplace=True)
        # print('df_container: ',df_container)
        # print('df_old: ',df_old)
        if not df_container.empty:
            df_container['machine_id'] = df_old['machine_id'].where(df_container['container_id'] == df_old['container_id'])
            # print('df_container2: ',df_container)
    df_node = df_container.groupby([df_container[it.tick_field], it.host_field],as_index=False).agg(it.dict_agg_metrics)
    node_details = df_node.values.tolist()
    container_details = df_container.values.tolist()
    # print("container_details",container_details)
    # print("node_details",node_details)
    
    z = {}
    # # print('node_details',node_details)
    first = True
    for n in node_details:
        curr_time = n[0]
        machine_id = n[1]
        cpu = n[2]
        if first:
            z[curr_time] = {
                                    "nodes": [{
                                    "machine_id": machine_id,
                                    "cpu": cpu
                                }], 
                                "containers": []
                                
                            }
            first = False
        else:
            y  = {
                            "machine_id": machine_id,
                            "cpu": cpu
                         }
            z[curr_time]['nodes'].append(y)

    first = True
    for r in container_details:
        curr_time = r[0]
        container_id = r[1]
        machine_id = r[2]
        cpu = r[3]
        # mem = r[4]
        
        y  = {
                        "container_id": container_id,
                        "machine_id": machine_id,
                        "cpu": cpu
                        }
        z[curr_time]['containers'].append(y)
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
