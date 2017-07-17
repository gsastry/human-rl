# Human Feedback

### Minimal usage

First run human_feedback:
Note the flag for online mode.

    python scripts/human_feedback.py --online --label_mode block -f /tmp/roadrunner/episodes/w0 -o /tmp/roadrunner/labels

This will start the human feedback interface and have it wait new frames to roll in.

Next, run A3C in online mode (in universe-starter-agent):

    python train.py --num-workers=1 --online --env-id RoadRunner --blocking_mode action_pruning --catastrophe_reward 100.0 --reward_scale 100.0 --log-dir logs/RoadRunnerOnline && sleep 3 && less logs/RoadRunnerOnline/logs/w-0.txt


Frames are sent to the human feedback interface, and wait for either a 'block' message (key 'b') or no message
to continue. An action that's supplied in the message tells the agent which action to take in case
the action is blocked.

You should see the log of proposed actions in the `w-0.txt` logs, and the log of what the feedback script
is doing in the human_feedback window. I generally run these side by side, and use the buttons: '3' for pause,
and 'b' for label.

### Notes

The online mode only works for episodes at a time (not intermediate frames).
After an episode is done and saved, it will ask you:

    Proceed to the next episode?

Entering anything other than 'y' in stdin will exit the viewer. At this point you can just ctrl-c the agent.
If you type 'y', it will proceed to the next episode.


### TODOs
#### Usability
#### Cleanup/refactor
#### Other envs than pong
