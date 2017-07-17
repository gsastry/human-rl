#!/bin/zsh
source activate humanrl
cd /home/william/code/human-rl/scripts
python human_feedback.py --online --label_mode block -i 4.0 
