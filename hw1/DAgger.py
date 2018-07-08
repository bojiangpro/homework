#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/RoboschoolHumanoid-v1.py --render \
            --num_rollouts 20
"""

import argparse
import pickle
import tensorflow as tf
import numpy as np
import gym
import importlib
import json
import io
from hw1.behavior_clone import BehaviorClone
import threading
import queue


def run(expert_policy_file, max_timesteps, num_rollouts=20, render=True, ):

    print('loading expert policy')
    module_name = expert_policy_file.replace('/', '.')
    if module_name.endswith('.py'):
        module_name = module_name[:-3]
    policy_module = importlib.import_module(module_name)
    print('loaded')

    env, policy = policy_module.get_env_and_policy()
    max_steps = max_timesteps or env.spec.timestep_limit

    # load training expected data
    module_dir = module_name.replace('.', '/')
    data_file = module_dir+'.json'
    fp = io.open(data_file, 'r')
    expert_data = json.load(fp)
    fp.close()
    # load pre trained behavior clone model
    behavior_clone = BehaviorClone(expert_data['observationSpace'], expert_data['actionSpace'],
                                   mode_directory=module_dir)

    for i in range(0, 10):
        m_return, std_return, observations, actions = data_agg(behavior_clone, num_rollouts, env, policy, max_steps)
        print('return', m_return)
        behavior_clone.learn(observations, actions)


def data_agg(behavior_clone, num_rollouts, env, policy, max_steps):
    returns = []
    observations = []
    actions = []
    obs_queue = queue.Queue(10)
    finished = False

    def gen():
        while not finished:
            yield np.reshape(np.array(obs_queue.get()), [1, -1])
        raise StopIteration()

    def input_fn():
        ds = tf.data.Dataset.from_generator(gen, tf.float64, tf.TensorShape([1,15]))
        return ds

    generator = behavior_clone.predict(input_fn)

    for i in range(num_rollouts):
        print('iter', i)
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            observations.append(obs)
            obs_queue.put(obs.tolist())
            b_action = next(generator)
            obs, r, done, _ = env.step(b_action)
            action = policy.act(obs)
            actions.append(action)
            totalr += r
            steps += 1
            if steps % 100 == 0:
                print("%i/%i" % (steps, max_steps))
            if steps >= max_steps:
                break
        returns.append(totalr)
    finished = True

    return np.mean(returns), np.std(returns), observations, actions


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    run('experts/RoboschoolHopper-v1.py', 1000)
