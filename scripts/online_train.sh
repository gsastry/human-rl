#!/bin/zsh
source activate humanrl
cd /home/william/code/human-rl/universe-starter-agent
python train.py --num-workers=1 --online True \
 --online_blocking_mode action_pruning && sleep 3 \
&& tail -f /tmp/pong/logs/w-0.txt ; tmux kill-session -t a3c
