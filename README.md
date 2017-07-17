# Human intervention reinforcement learning 

Disclaimer: this is research code that has only been tested on a couple of platforms.

Contributors (alphabetical): Owain Evans, Vlad Firoiu, Girish Sastry, William Saunders


## Overview

This repository contains the code for human intervention reinforcement learning in Atari environments (based on OpenAI's Gym). The `humanrl` package contains various Gym environment wrappers and utilities that allow modifying Atari environments to include catastrophes.

`scripts/human_feedback.py` is a script that allows a human to intervene during offline or online training of an RL agent.

## Installation and use

To label and run the code locally, first create an Anaconda environment with our packages:

```bash
conda env create
source activate humanrl
```

See [the human feedback README](https://github.com/gsastry/human-rl/tree/master/scripts/README.md) for directions on providing human feedback with the OpenAI universe starter agent.

See the [catastrophe wrapper](https://github.com/gsastry/human-rl/blob/master/humanrl/catastrophe_wrapper.py) for a general purpose way to add catastrophes to Gym environments.

