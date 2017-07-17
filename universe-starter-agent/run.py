import os
import go_vncdriver
import tensorflow as tf
import argparse
import json
import envs
from model import policies
import checkpoint_utils

parser = argparse.ArgumentParser(description="Run commands")
parser.add_argument('logdir', type=str, help="Log directory path")

args = parser.parse_args()

with open(args.logdir + "/hparams.json") as f:
    hparams = json.load(f)

env = envs.create_env(**hparams)
obs = env.reset()

policyType = policies[hparams['policy']]
policy = policyType(env.observation_space.shape, env.action_space.n, **hparams)
features = policy.get_initial_features()

sess = tf.Session()

#import ipdb; ipdb.set_trace()

checkpoint_utils.init_from_checkpoint(args.logdir + '/train', {'global/':'/'})
#saver = tf.train.Saver(sharded=True)
#saver.restore(sess, os.path.join(args.logdir, 'train/model.ckpt-0'))
sess.run(tf.global_variables_initializer())

with sess.as_default():
    
    while True:
        env.render()
        
        fetched = policy.act(obs, *features)
        action, value_, features = fetched[0], fetched[1], fetched[2:]

        obs, reward, done, info = env.step(action.argmax())
        
        if done:
            obs = env.reset()
