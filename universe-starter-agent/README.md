# universe-starter-agent for human-in-loop

## Run A3C Experiments with Catastrophe Wrappers

### No penalties/blocking (just saving frames)
To run A3C without any catastrophe penalties/blocking:

`python train.py --num-workers 16 --env-id Pong --log-dir $log_dir --catastrophe_reward 0`

The script `train.py` starts the workers (and is not modified from the original). This calls `worker.py` which creates a gym env and runs A3C on the env. The script `envs.py` is where the env is constructed and where catastrophe wrappers are added. In the above command (where we didn't set a `catastrophe_type` argument), the only wrapper used is `frame.FrameSaveWrapper`, which just saves the frames.

Note: in `envs.py`, the function `make_env` converts 'Pong' to 'PongDeterministic-v3' and does the same for the other games. The deterministic versions are easier. 

### Penalties using hand-coded catastrophe labeller
Penalties can be provided either by a hand-coded labeller or a TF classifier. For the Pong hand-coded labeller we use:

`python train.py --num-workers 16 --env-id Pong --log-dir $log_dir --catastrophe_reward -1 --catastrophe_type 1`

Since we used 'Pong' and `catastrophe_type=1`, the script `envs.py` specifies the classifier as `pong_catastrophe.CatastropheClassifierHeuristic`. This just uses the raw pixels to decide if the paddle is below the `DEFAULT_CLEARANCE` variable. This classifier is used (along with `catastrophe_reward`) as input to the `CatastropheWrapper` constructor. The `CatastropheWrapper` (in `catastrophe_wrapper.py`) checks whether the current observation is a catastrophe and supplies the `catastrophe_reward` in case it is. (It also does blocking and records differences between the main classifier and a baseline catastrophe classifier).

NOTE: I'm not sure of the difference between `pong_catastrophe.CatastropheClassifierHeuristic` and `pong_catastrophe.PongClassifierLabeller`. The latter is used in `test_pipeline.ipy`. 

### Penalties using TF catastrophe classifier
Having saved frames using 'No Penalties" setting (above), we can label the frames and then train a classifier on the labels. This is mostly done in `classifier_tf.py`. We label a dataset using:

```python
data_loader = DataLoader(pong_catastrophe.PongClassifierLabeller(), TensorflowClassifierHparams(hparams))
datasets = data_loader.split_episodes(episode_paths, n_train, n_valid, n_test, use_all=False)
```

We run training as follows (actually trains a series of classifiers with different hparams):

`run_experiments(logdir, data_loader, datasets, common_hparams, hparams_list, steps, log_every=50)`

This will save a classifier with `path=logdir/0/final.ckpt`. To run a3c with a penalty provided by this classifier, we run:

`python train.py --num-workers 16 --env-id Pong --log-dir $logdir --catastrophe_reward -1  --classifier_file $path  --catastrophe_type 1`

See `test_pipeline.py` for details. 


### Blockers











-----------------------

-----------------------

----------------------


# universe-starter-agent README


The codebase implements a starter agent that can solve a number of `universe` environments.
It contains a basic implementation of the [A3C algorithm](https://arxiv.org/abs/1602.01783), adapted for real-time environments.

# Dependencies

* Python 2.7 or 3.5
* [six](https://pypi.python.org/pypi/six) (for py2/3 compatibility)
* [TensorFlow](https://www.tensorflow.org/) 0.11
* [tmux](https://tmux.github.io/) (the start script opens up a tmux session with multiple windows)
* [htop](https://hisham.hm/htop/) (shown in one of the tmux windows)
* [gym](https://pypi.python.org/pypi/gym)
* gym[atari]
* [universe](https://pypi.python.org/pypi/universe)
* [opencv-python](https://pypi.python.org/pypi/opencv-python)
* [numpy](https://pypi.python.org/pypi/numpy)
* [scipy](https://pypi.python.org/pypi/scipy)

# Getting Started

```
conda create --name universe-starter-agent python=3.5
source activate universe-starter-agent

brew install tmux htop      # On Linux use sudo apt-get install -y tmux htop

pip install gym[atari]
pip install universe
pip install six
pip install tensorflow
conda install -y -c https://conda.binstar.org/menpo opencv3
conda install -y numpy
conda install -y scipy
```


Add the following to your `.bashrc` so that you'll have the correct environment when the `train.py` script spawns new bash shells
```source activate universe-starter-agent```

## Atari Pong

`python train.py --num-workers 2 --env-id PongDeterministic-v3 --log-dir /tmp/pong`

The command above will train an agent on Atari Pong using ALE simulator.
It will see two workers that will be learning in parallel (`--num-workers` flag) and will output intermediate results into given directory.

The code will launch the following processes:
* worker-0 - a process that runs policy gradient
* worker-1 - a process identical to process-1, that uses different random noise from the environment
* ps - the parameter server, which synchronizes the parameters among the different workers
* tb - a tensorboard process for convenient display of the statistics of learning

Once you start the training process, it will create a tmux session with a window for each of these processes. You can connect to them by typing `tmux a` in the console.
Once in the tmux session, you can see all your windows with `ctrl-b w`.
To switch to window number 0, type: `ctrl-b 0`. Look up tmux documentation for more commands.

To access TensorBoard to see various monitoring metrics of the agent, open [http://localhost:12345/](http://localhost:12345/) in a browser.

Using 16 workers, the agent should be able to solve `PongDeterministic-v3` (not VNC) within 30 minutes (often less) on an `m4.10xlarge` instance.
Using 32 workers, the agent is able to solve the same environment in 10 minutes on an `m4.16xlarge` instance.
If you run this experiment on a high-end MacBook Pro, the above job will take just under 2 hours to solve Pong.

![pong](https://github.com/openai/universe-starter-agent/raw/master/imgs/tb_pong.png "Pong")

For best performance, it is recommended for the number of workers to not exceed available number of CPU cores.

You can stop the experiment with `tmux kill-session` command.
