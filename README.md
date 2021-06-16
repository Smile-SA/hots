# **cots**

> Application for testing a hybrid resource allocation method using machine learning and optimization.

This is a version for testing COTS using Docker containers

# current issue to fix in the cots package --> cots crashing at stage 'Building relaxed model'
---------------------------------

## Requirements for running


### Install docker on local machine

Installation steps:
https://docs.docker.com/engine/install/ubuntu/

### Install docker-compose

Installation steps:
https://docs.docker.com/compose/install/

##Known issues:
On running docker there may be issues with connection denied.

This page proposes the remedy to this problem:
https://docs.docker.com/engine/install/linux-postinstall/

run the following commands:
```bash
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker
```

### Check the Dockerfile to see if the uid is 1000, change otherwise

check the uid of the local machine user
```bash
more /etc/passwd
```
if the uid of the user is 1000 then no need to edit the Dockerfile
Else edit the Dockerfile
look for the comments 'Replace 1000 with your user / group id'
Change 1000 to your uid and gid from the /etc/passwd file

### Running the COTS testing via docker

Run the following commands in a bash shell at the root of the directory where ther is the Dockerfile:

# build the cots image
```bash
docker build -t cots
```

# run the cots image in a container with export of display
```bash
docker run -ti --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix cots
```

This last command will run the cots image and drop you in a bash shell of the cots container
At the shell run the cots testing with the command:

# run cots testing
```bash
cots --data /rac/tests/data/generated_30 --params /rac/tests/data/params.json 
```
#outputs
terminal output and matplotlib graphs 

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
