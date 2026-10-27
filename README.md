<h1 align="center">
<img src="https://github.com/Smile-SA/hots/raw/main/doc/source/_static/hots_logo.png" width="250">
</h1><br>

> Hybrid Optimization for Time Series (HOTS)  
> HOTS solves problems presented as time series using machine learning and optimization methods.  
> The library supports multiple resource related problems (placement, allocation), presented as one or more metrics.

## Requirements for running HOTS

HOTS works on any platform with Python 3.10 and up.

The dev Python version must be used to install the package (for example install the package
python3.10-dev in order to use Python 3.10).

A solver needs to be installed for using HOTS. HOTS uses GLPK by default, which is not a Python
package: install it with your system's package manager before using HOTS (on Debian or Ubuntu,
`sudo apt install glpk-utils libglpk-dev`).

In order to run HOTS with the streaming platform Kafka, you need the `kafka` extra (see below),
and a running Kafka broker.

## Installing HOTS

From PyPI:

```bash
pip install hots            # file connector only
pip install "hots[kafka]"   # with the Kafka connector
hots --version
```

For development, a Makefile is provided, which creates a virtual environment and installs HOTS
from the checkout. You can do :

```bash
make
```

## Configuration file

HOTS is configured via a single JSON file that you pass to the CLI.

A minimal example using the file connector looks like this:

```json
{
  "time_limit": null,
  "data": {
    "tick_field": "timestamp",
    "individual_field": "container_id",
    "host_field": "machine_id",
    "metrics": ["cpu"]
  },
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
      "tick_increment": 2,
      "window_duration": 3,
      "sep_time": 3,
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

The `data` section names the columns of your time series. These describe the
*data*, not the transport, so every component reads them from there and you can
swap connectors without touching them. Only one entry in `metrics` is supported
today — the pipeline optimizes a single metric, and a longer list is rejected at
load time rather than silently ignored.

> **Backward compatibility:** these four keys may still be given inside
> `connector.parameters` instead, as in older configs; they are then used to
> populate `data` automatically. Declaring the same key in both places with
> *different* values is a config error.

For a Kafka-based setup, the structure is the same but the `connector` section uses `"type": "kafka"`
(see `hots/config/sample_config_kafka.json`):

```json
"connector": {
  "type": "kafka",
  "parameters": {
    "bootstrap.servers": "localhost:9092",
    "topic_data": "hots_data",
    "topic_moves": "hots_moves",
    "consumer_conf": {
      "bootstrap.servers": "localhost:9092",
      "group.id": "hots",
      "auto.offset.reset": "earliest"
    },
    "tick_increment": 10,
    "window_duration": 3,
    "sep_time": 30,
    "batch_size": 10,
    "poll_timeout": 1.0,
    "bootstrap_timeout_seconds": 120,
    "node_capacities": {"vm1": 2.0, "vm2": 2.0, "vm3": 2.0}
  }
}
```

| Parameter | Description |
| --- | --- |
| `bootstrap.servers` | Kafka broker HOTS publishes moves to. |
| `topic_data` | Topic HOTS reads telemetry from. |
| `topic_moves` | Topic HOTS publishes its moves to. |
| `consumer_conf` | [librdkafka consumer settings](https://github.com/confluentinc/librdkafka/blob/master/CONFIGURATION.md) for reading `topic_data`; `group.id` is required. |
| `sep_time` | HOTS reads telemetry until it reaches this tick before it computes its first placement. |
| `batch_size` | A streaming iteration stops reading once it has this many telemetry rows (one row per container sample). |
| `poll_timeout` | Seconds to wait for a message. During the run, an iteration that receives nothing in this time **ends the run**: set it above the producer's publish interval. |
| `bootstrap_timeout_seconds` | Seconds to wait for the first `sep_time` ticks before failing. Default 120. |
| `node_capacities` | Capacity of each host for the metric in `data.metrics`. When absent, the largest observed load of each host during bootstrap is used instead. |

See the User manual in the documentation for the other sections.

## Running HOTS on a live environment

The Kafka connector exchanges two kinds of messages. It reads **telemetry** batches,
`{"v": 1, "containers": [{"timestamp", "container_id", "machine_id", "<metric>"}, ...]}`, from
`topic_data`, and writes one **move** per message,
`{"v": 1, "container_name", "old_host", "new_host", "time"}`, to `topic_moves`. The examples in
`tests/contracts/fixtures/` define this contract.

HOTS doesn't observe or move workloads itself:

- [hots-connector](https://github.com/Smile-SA/hots-connector) produces the telemetry from
  Prometheus and applies the moves on Docker or Kubernetes;
- the [mums benchmark platform](https://github.com/Smile-SA/mums) runs HOTS and hots-connector
  together against a Kubernetes cluster, and compares the result with other placement methods.

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

Example configs are provided in the package (`hots/config/sample_config_file.json` and
`hots/config/sample_config_kafka.json`). The small dataset the file example reads,
`tests/data/thesis_ex_10`, is in the source repository, and the path is relative to the working
directory. From a checkout, you can quickly test your installation with:

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
