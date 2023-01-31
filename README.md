# **HOTS**

> Application for testing a hybrid resource allocation method using machine learning and optimization.

This is a version for testing HOTS in a Docker containers
This version includes test data with all nodes

# current issue to fix in the hots package --> crashing on data with all nodes

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

### Running the HOTS testing via docker

Run the following commands in a bash shell at the root of the directory where ther is the Dockerfile:

# build the hots image
```bash
docker build -t hots .
```

# run the hots image in a container with export of display
```bash
docker run -ti --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix hots
```

This last command will run the hots image and drop you in a bash shell of the hots container
You will be at /home/developer which is empty

# check the content of /rac where the repo has been copied
```bash
ls /rac
```


At the shell run the hots testing with the command:

# run hots testing on test data
```bash
hots /rac/tests/data/generated_7
```
<!-- # run hots testing on test data with all nodes
```bash
hots --data /rac/tests/alibaba_short_time_interval_test_data --params /rac/tests/alibaba_short_time_interval_test_data/params.json 
``` -->


#outputs
terminal output and matplotlib graphs 

## Credits

This software is sponsored by [Smile](https://www.smile.fr/).

The team:

- Jonathan Rivalan - Project manager
- Etienne Leclercq - Software design, lead developer
- Marco Mariani
- Gilles Lenfant

## Links

- [Project home](https://git.rnd.smile.fr/overboard/soft_clustering/rac)
- [File issues (bugs, ...)](https://git.rnd.smile.fr/overboard/soft_clustering/rac/-/issues)

## License

This software is provided under the terms of the MIT license you can read in the `LICENSE.txt` file
of the repository or the package.
