#!/bin/sh
docker build -t base -f base.docker .
docker build -t main -f main.docker .
