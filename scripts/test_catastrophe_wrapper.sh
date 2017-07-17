#! /bin/bash
source activate humanrl
cd /home/william/code/human-rl/universe-starter-agent
rm -rf /tmp/pong; python train.py --catastrophe_type 1 --classifier_file models/pong/c1/classifier/final.ckpt --blocker_file models/pong/b1/0/final.ckpt --blocking_mode action_pruning --catastrophe_reward -1 && sleep 3 && tail -f /tmp/pong/logs/w-0.txt
