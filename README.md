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
hots /path/to/data/folder
```

Make sure to activate the virtual environment before running HOTS with :

```bash
source venv/bin/activate
```

Some parameters can be defined with the `hots` command, such as :
 * `k` : the number of clusters used in clustering ;
 * `tau` : the window size during the loop process ;
 * `param` : a specific parameter file to use.

All the CLI options are found running the `help` option :
```bash
hots --help
```

More parameters can be defined through a `.JSON` file, for which an example is provided in the `tests` folder. See the documentation, section `User manual`, for more details about all the parameters.  

Note that a test data is provided within the package, so you can easily test the installation with :
```bash
hots /tests/data/generated_7
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
