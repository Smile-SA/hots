<h1 align="center">
<img src="https://github.com/Smile-SA/hots/raw/main/doc/source/_static/hots_logo.png" width="250">
</h1><br>

> Hybrid Optimization for Time Series (HOTS)  
> HOTS solves problems presented as time series using machine learning and optimization methods.  
> The library supports multiple resource related problems (placement, allocation), presented as one or more metrics.

## Requirements for running HOTS

HOTS works on any platform with Python 3.8 and up.

The dev Python version must be used to install the package (for example install the package
python3.10-dev in order to use Python 3.10).

A solver needs to be installed for using HOTS. By default, GLPK is installed with HOTS, but the
user needs to install the following packages before using HOTS :
 * libglpk-dev
 * glpk-utils

In order to run HOTS with the streaming platform Kafka, you need to install the
Python package confluent_kafka, and to have a running kafka broker.

## Installing HOTS

A Makefile is provided, which creates a virtual environment and install HOTS. You can do :

```bash
make
```

## Configuration file

HOTS is configured via a single JSON file that you pass to the CLI.

A minimal example using the file connector looks like this:

```json
{
  "time_limit": null,
  "clustering": {
    "method": "kmeans",
    "nb_clusters": 3,
    "parameters": {
      "nb_clusters": 3
    }
  },
  "optimization": {
    "backend": "pyomo",
    "parameters": {
      "solver": "glpk",
      "verbose": 0
    }
  },
  "problem": {
    "type": "placement",
    "parameters": {
      "initial_placement": 0,
      "tol": 0.1,
      "tol_move": 0.5
    }
  },
  "connector": {
    "type": "file",
    "parameters": {
      "data_folder": "./tests/data/thesis_ex_10",
      "file_name": "container_usage.csv",
      "host_field": "machine_id",
      "individual_field": "container_id",
      "tick_field": "timestamp",
      "tick_increment": 2,
      "window_duration": 3,
      "sep_time": 3,
      "metrics": ["cpu"],
      "outfile": "./out/moves.log"
    }
  },
  "logging": {
    "level": "INFO",
    "filename": "./out/hots.log",
    "fmt": "%(asctime)s %(levelname)s: %(message)s"
  },
  "reporting": {
    "results_folder": "./out",
    "metrics_file": "./out/metrics.csv",
    "plots_folder": "./out/plots"
  }
}
```

For a Kafka-based setup, the structure is the same but the `connector` section uses `"type": "kafka"` and Kafka-specific parameters:

```json
"connector": {
  "type": "kafka",
  "parameters": {
    "connector_url": "http://localhost:8080",
    "bootstrap.servers": "localhost:9092",
    "topics": ["hots_moves"],
    "data_folder": "./tests/data/thesis_ex_10",
    "file_name": "container_usage.csv",
    "host_field": "machine_id",
    "individual_field": "container_id",
    "tick_field": "timestamp",
    "tick_increment": 3,
    "window_duration": 3,
    "sep_time": 10,
    "metrics": ["cpu"],
    "outfile": "./out/moves.log"
  }
}
```
See the User manual in the documentation for a full description of each field.

## Configuring Kafka

Before running the application, you need to configure the Kafka broker information and the topic name in the `params.json` file. Open the file and make the following changes:
```json
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

## Running HOTS

### Via Python

The application can be used simply by running :

```bash
hots --config path/to/config.json
```

Make sure to activate the virtual environment before running HOTS with :

```bash
source venv/bin/activate
```

The only mandatory argument is the path to the JSON configuration file.
You can inspect available options with:

```bash
hots --help
```

A small test dataset and example configs are provided in the package (see `hots/config/sample_config_file.json` and `hots/config/sample_config_kafka.json`).
You can quickly test your installation with:

```bash
hots --config hots/config/sample_config_file.json
```

### Via Docker

A Docker container can be easily built for running `hots`, using the `Dockerfile` provided in the package.
If you are not used to Docker, you can follow the installation guideline here : https://docs.docker.com/engine/install/, and the post-install process here (Linux) : https://docs.docker.com/engine/install/linux-postinstall/.

As soon as Docker is setup, you can run the following commands (being at the root of the directory, with the Dockerfile) :

```bash
docker build -t hots .
```

Once the container is created, you can run it, by running the following :

```bash
docker run -it hots /bin/bash
```

You will be prompted to a new shell, in which you can follow the same steps as for Python.

## Credits

Authors:

- Etienne Leclercq - Software design, lead developer
- Jonathan Rivalan - Product owner, Lead designer 
- Mufaddal Enayath Hussain
- Marco Mariani
- Gilles Lenfant
- Soobash Daiboo
- Kang Du
- Amaury Sauret
- SMILE R&D

As HOTS was created during a PhD, credits have to be given to academic supervisors, Céline Rouveirol and Frédéric Roupin, involved in the algorithm thinking.

## Links

- [Project home](https://github.com/Smile-SA/hots)
- [File issues (bugs, ...)](https://github.com/Smile-SA/hots/issues)
- [PyPi package](https://pypi.org/project/hots/)
- [Documentation](https://hots.readthedocs.io/en/latest/)
- [PhD document](https://theses.hal.science/tel-03997934)

## License

This software is provided under the terms of the MIT license you can read in the `LICENSE.txt` file of the repository or the package.
