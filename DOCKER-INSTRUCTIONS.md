# Docker container
To use, first install docker: https://docs.docker.com/engine/installation/

To build and start the docker image:

```
docker build -t base -f base.docker .
docker build -t main -f main.docker .
```

On Ubuntu:
```
docker run --name=human-rl -t -i -v /var/run/docker.sock:/var/run/docker.sock --net=host -v `pwd`:/mnt/human-rl/ main
```

On OS X (works on 10.12.2):
```
docker run --privileged -p 5901:5900 -v /usr/bin/docker:/user/bin/docker -v /var/run/docker.sock:/var/run/docker.sock -v `pwd`:/mnt/human-rl -e DOCKER_NET_HOST=172.17.0.1 -t -i main
open vnc://localhost:5901
```

Which launches a command line version of the docker container

and to restart the docker container later:

`docker start human-rl`

`docker attach human-rl`

(Note: the -v /var/run/docker.sock:/var/run/docker.sock --net=host options are necessary to allow the universe to use automatic remotes. This may not work outside of ubuntu. In this case, you may need to manually start universe remotes and point openai gym at them, see https://github.com/openai/universe/blob/master/doc/remotes.rst#how-to-start-a-remote)

It also opens a vnc server on port 5900. To view gym environments, you can run the training from the vnc session. (password is openai)

GPU support seems to work. One necessary prerequisite is https://github.com/NVIDIA/nvidia-docker. (For ubuntu 16.10, the following fix is required: https://github.com/NVIDIA/nvidia-docker/issues/234)
