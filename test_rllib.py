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


EXAMPLE_USAGE = """
example usage:
    python ./visualizer_rllib.py /ray_results/experiment_dir/result_dir 1

Here the arguments are:
1 - the path to the simulation results
2 - the number of the checkpoint
"""

def visualizer_rllib(args):
    """Visualizer for RLlib experiments.

    This function takes args (see function create_parser below for
    more detailed information on what information can be fed to this
    visualizer), and renders the experiment associated with it.
    """
    if (not ray.is_initialized()):
        ray.init(num_cpus=1)
    result_dir = args.result_dir if args.result_dir[-1] != '/' \
        else args.result_dir[:-1]

    config = get_rllib_config(result_dir)

    # Run on only one cpu for rendering purposes
    config['num_workers'] = 0

    flow_params = get_flow_params(config)

    # hack for old pkl files
    # TODO(ev) remove eventually
    sim_params = flow_params['sim']
    print("OBSERVATION DELAY ", flow_params['env'].additional_params['observation_delay'])
    setattr(sim_params, 'num_clients', 1)

    # for hacks for old pkl files TODO: remove eventually
    if not hasattr(sim_params, 'use_ballistic'):
        sim_params.use_ballistic = False

    # Determine agent and checkpoint
    config_run = config['env_config']['run'] if 'run' in config['env_config'] \
        else None
 
    agent_cls = get_agent_class(config_run)

    sim_params.restart_instance = True
    dir_path = os.path.dirname(os.path.realpath(__file__))
    emission_path = '{0}/test_time_rollout/'.format(dir_path)
    sim_params.emission_path = None

    # Create and register a gym+rllib env
    create_env, env_name = make_create_env(params=flow_params, version=0)
    register_env(env_name, create_env)

    env_params = flow_params['env']
    env_params.restart_instance = False

    # lower the horizon if testing
    if args.horizon:
        config['horizon'] = args.horizon
        env_params.horizon = args.horizon

    # create the agent that will be used to compute the actions
    agent = agent_cls(env=env_name, config=config)
    checkpoint = result_dir + '/checkpoint_' + args.checkpoint_num
    checkpoint = checkpoint + '/checkpoint-' + args.checkpoint_num
    agent.restore(checkpoint)

    env = gym.make(env_name)

    if args.render_mode == 'sumo_gui':
        env.sim_params.render = True  # set to True after initializing agent and env
        if args.save_render:
            env.sim_params.save_render = True
            env.path = "/home/alex/flow_video_rl_4_delay"
    # if restart_instance, don't restart here because env.reset will restart later
    if not sim_params.restart_instance:
        env.restart_simulation(sim_params=sim_params, render=sim_params.render)

    # Simulate and collect metrics
    vel = []
    state = env.reset()
    ret = 0
    for _ in range(env_params.horizon):
        vehicles = env.unwrapped.k.vehicle
        speeds = vehicles.get_speed(vehicles.get_ids())

        # only include non-empty speeds
        if speeds:
            vel.append(np.mean(speeds))

        action = agent.compute_action(state)
        state, reward, done, _ = env.step(action)
        ret += reward
        if done:
            break
    mean_speed = np.mean(vel)
    std_speed = np.std(vel)

    print('==== Summary of results ====')
    print("Return:")
    print(ret)

    print(f"\nSpeed, mean (m/s):{mean_speed}")
    print(f"\nSpeed, std (m/s):{std_speed}")

    result = [ret, mean_speed, std_speed]
    # terminate the environment
    env.unwrapped.terminate()
    return result

def create_parser():
    """Create the parser to capture CLI arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='[Flow] Evaluates a reinforcement learning agent '
                    'given a checkpoint.',
        epilog=EXAMPLE_USAGE)

    # required input parameters
    parser.add_argument(
        'result_dir', type=str, help='Directory containing results')
    parser.add_argument('checkpoint_num', type=str, help='Checkpoint number.')

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
    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    result = joblib.Parallel(n_jobs=10, backend="multiprocessing")(delayed(visualizer_rllib)(args) for _ in range(args.num_rollouts))
    print(result)
