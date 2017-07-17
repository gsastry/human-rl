#!/bin/bash
# _mapping SpaceInvaders

LOGDIR="logs/SpaceInvadersOnline2"

ls "${LOGDIR}/episodes/w0" || echo "No episodes exist"

if [ "$1" = "train" ] ; then
  tmux kill-session -t a3c || true
  cd universe-starter-agent
  python train.py --num-workers=1 --online True --env-id SpaceInvaders \
  --online_blocking_mode action_replacement --catastrophe_reward 0 --reward_scale 1.0 \
  --log-dir "${LOGDIR}" --max_episodes None --render False \
  --extra_wrapper SpaceInvadersStartsWrapper2
  sleep 3
  tail -f "${LOGDIR}/logs/w-0.txt"
elif [ "$1" = "human_feedback" ] ; then
  pkill -f human_feedback.py || true
  python scripts/human_feedback.py --online --label_mode block --safe_action_mapping SpaceInvaders --pause -i 4.0
fi
