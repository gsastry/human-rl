#!/bin/sh
sed 's/tensorflow:latest-py3/tensorflow:latest-gpu-py3/' base.docker > base-gpu.docker
docker build -t base -f base-gpu.docker .
docker build -t main -f main.docker .
