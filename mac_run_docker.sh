docker run --privileged -p 5901:5900 -p 12345:12345 -v /usr/bin/docker:/user/bin/docker -v /var/run/docker.sock:/var/run/docker.sock -v `pwd`:/mnt/human-rl -e DOCKER_NET_HOST=172.17.0.1 -t -i main
