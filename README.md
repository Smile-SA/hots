# **cots**

> Application for testing a hybrid resource allocation method using machine learning and optimization.

This is a version for testing COTS using Docker containers

## Requirements for running

Once installation:
Install docker on local machine

For Ubuntu the following page lists the installation steps:
https://docs.docker.com/engine/install/ubuntu/

Once installation:
Install docker-compose

For Ubuntu the folowing page lists the installation steps:
https://docs.docker.com/compose/install/

##Known issues:
On running docker there may be issues with connection denied.

This page proposes the remedy to this problem:
https://docs.docker.com/engine/install/linux-postinstall/

run the following commands:
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker

### Running the COTS testing via docker

Run the following commands in a bash shell at the root of the directory where ther is the Dockerfile:

docker-compose build

docker-compose up

docker run -ti cots

This last command will run the cots image and drop you in a bash shell of the cots container
At the shell run the cots testing with the command:

cots --data /rac/tests/data/generated_30 --params /rac/tests/data/params.json 



## Credits

This software is sponsored by [Alter Way](https://www.alterway.fr/).

The team:

- Jonathan Rivalan - Project manager
- Etienne Leclercq - Software design, lead developer
- Marco Mariani
- Gilles Lenfant

## Links

- [Project home](https://git.rnd.alterway.fr/overboard/soft_clustering/rac)
- [File issues (bugs, ...)](https://git.rnd.alterway.fr/overboard/soft_clustering/rac/-/issues)

## License

This software is provided under the terms of the MIT license you can read in the `LICENSE.txt` file
of the repository or the package.
