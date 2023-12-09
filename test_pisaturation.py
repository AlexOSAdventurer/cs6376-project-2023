import argparse
import gym
import numpy as np
import os
import sys
import time
import joblib
from joblib import Parallel, delayed

import ray
try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class
from ray.tune.registry import register_env

from flow.core.util import emission_to_csv
from flow.utils.registry import make_create_env
from flow.utils.rllib import get_flow_params
from flow.utils.rllib import get_rllib_config
from flow.utils.rllib import get_rllib_pkl
from ring_delay_pi import generateConfig
import traceback


def visualizer_pisaturation(args):
    flow_params = generateConfig(args.delay)

    sim_params = flow_params['sim']
 
    sim_params.restart_instance = True
    sim_params.emission_path = None

    try:
        # Create and register a gym+rllib env
        create_env, env_name = make_create_env(params=flow_params, version=0)
        register_env(env_name, create_env)

        env_params = flow_params['env']
        env_params.restart_instance = False

        # lower the horizon if testing
        if args.horizon:
            env_params.horizon = args.horizon
        env = create_env()
    except Exception as e:
        traceback.print_exc()
        raise e

    if args.render_mode == 'sumo_gui':
        env.sim_params.render = True  # set to True after initializing agent and env
        if args.save_render:
            env.sim_params.save_render = True
            env.path = "/home/alex/flow_video_pisaturation_5_delay"
    # if restart_instance, don't restart here because env.reset will restart later
    if not sim_params.restart_instance:
        env.restart_simulation(sim_params=sim_params, render=sim_params.render)

    # Simulate and collect metrics
    vel = []
    monitor_state = []
    state = env.reset()
    ret = 0
    for _ in range(env_params.horizon):
        vehicles = env.unwrapped.k.vehicle
        speeds = vehicles.get_speed(vehicles.get_ids())

        # only include non-empty speeds
        if speeds:
            vel.append(np.mean(speeds))
    
        state, reward, done, _ = env.step(None)
        ret += reward
        monitor_state.append(env.evaluate_safety())
        if done:
            break
    mean_speed = np.mean(vel)
    std_speed = np.std(vel)

    percentage_safety = np.mean(monitor_state)

    print('==== Summary of results ====')
    print("Return:")
    print(ret)

    print(f"\nSpeed, mean (m/s):{mean_speed}")
    print(f"\nSpeed, std (m/s):{std_speed}")

    print(f"\nPercentage safety:{percentage_safety}")

    result = [ret, mean_speed, std_speed, percentage_safety]
    # terminate the environment
    env.unwrapped.terminate()
    return result

def create_parser():
    """Create the parser to capture CLI arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='[Flow] Evaluates a reinforcement learning agent '
                    'given a checkpoint.',
        epilog="")

    # optional input parameters
    parser.add_argument(
        '--num_rollouts',
        type=int,
        default=1,
        help='The number of rollouts to visualize.')
    parser.add_argument(
        '--render_mode',
        type=str,
        default='sumo_gui',
        help='Pick the render mode. Options include sumo_web3d, '
             'rgbd and sumo_gui')
    parser.add_argument(
        '--save_render',
        action='store_true',
        help='Saves a rendered video to a file. NOTE: Overrides render_mode '
             'with pyglet rendering.')
    parser.add_argument(
        '--horizon',
        type=int,
        help='Specifies the horizon.')
    parser.add_argument(
        '--delay',
        type=float,
        default=0.0,
        help="Delay in seconds."
    )
    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    result = joblib.Parallel(n_jobs=10, backend="multiprocessing")(delayed(visualizer_pisaturation)(args) for _ in range(args.num_rollouts))
    #visualizer_pisaturation(args)
    print(result)
