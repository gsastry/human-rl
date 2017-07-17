export PYTHONPATH=/mnt/human-rl

if ! pgrep -x "x11vnc" > /dev/null ; then
  /usr/bin/x11vnc -forever -usepw -create &
fi
if ! pgrep -x "jupyter" > /dev/null ; then
  cd /mnt/human-rl/notebooks
  PYTHONPATH="/mnt/human-rl/universe-starter-agent:$PYTHONPATH" jupyter notebook --NotebookApp.token='21424b3059e278f2525c403395540d13f0f4afd9247b22a521424b3059e278f2525c403395540d13f0f4afd9247b22a5' --allow-root &
  echo "Running at localhost:8888"
  echo "Token is 21424b3059e278f2525c403395540d13f0f4afd9247b22a521424b3059e278f2525c403395540d13f0f4afd9247b22a5"
  cd -
fi

bind '"\e[A": history-search-backward'
bind '"\e[B": history-search-forward'
