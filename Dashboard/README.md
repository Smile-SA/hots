# HOTS with Apache Kafka!

This README will guide you through the setup and usage of this version of the code that has integrated Apache Kafka for data streaming.

## Requirements

Apart from the components needed to run the Original version of HOTS, some added requirements are necessary when running this version.

1. Install the required dependencies in python (main python package used is confluent_kafka)
2. A running kafka broker

##  Configuring Kafka

Before running the application, you need to configure the Kafka broker information and the topic name in the `param_default.json` file. Open the file and make the following changes:
```
"kafkaConf":{
 "topics":{
  "docker_topic": "xxxxx"
 },
 "Producer":{
  "brokers":["<IP>:9092"]
 },
 "Consumer":{
  "group": "xxxx",
  "brokers":["<IP>:9092"]
 }
}
```
Additionally, you may need to make necessary changes in the `kafka.py` file to match your Kafka setup. Ensure that the Kafka producer and consumer configurations are correctly set.
At present, features such as schema registry and SSL encryption are disabled by default.

## Data Streaming
To send real-time data from an external file to HOTS via Kafka, you can use the provided `data_stream.py` script. This script will produce data to the Kafka topic specified in the `param_default.json` file. However, please ensure that the data in the streamed file is a continuation of the historical data. The timestamps should be consistent, and the streamed file should start where the historical data ends.

HOTS will receive the real-time data and perform the next steps of the analysis as the data is sent to it.

Once the HOTS application has consumed all the data from the Kafka topic and finished its analysis, it will automatically exit and behave the same way as the original HOTS version.