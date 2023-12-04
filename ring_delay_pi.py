from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import VehicleParams, SumoCarFollowingParams
from flow.controllers import RLController, IDMController, ContinuousRouter
from wave_attenuation import WaveAttenuationPOEnvWithDelay
from non_rl_controllers import PISaturationWithDelay
from flow.networks import RingNetwork
import os

def generateConfig(delay):
    # time horizon of a single rollout
    HORIZON = 3000

    # We place one autonomous vehicle and 22 human-driven vehicles in the network
    vehicles = VehicleParams()
    vehicles.add(
        veh_id="human",
        acceleration_controller=(IDMController, {
            "noise": 0.2
        }),
        car_following_params=SumoCarFollowingParams(
            min_gap=0
        ),
        routing_controller=(ContinuousRouter, {}),
        num_vehicles=21)
    vehicles.add(
        veh_id="rl",
        acceleration_controller=(PISaturationWithDelay, {}),
        routing_controller=(ContinuousRouter, {}),
        num_vehicles=1)

    flow_params = dict(
        # name of the experiment
        exp_tag=f"stabilizing_the_ring_pi_delay_{delay}",
        env_name=WaveAttenuationPOEnvWithDelay,
        network=RingNetwork,
        simulator='traci',
        sim=SumoParams(
            sim_step=0.1,
            render=False,
            save_render=False,
            restart_instance=False,
        ),
        env=EnvParams(
            horizon=HORIZON,
            warmup_steps=750,
            clip_actions=False,
            additional_params={
                "max_accel": 1,
                "max_decel": 1,
                "ring_length": [220, 270],
                "observation_delay": delay,
            },
        ),
        net=NetParams(
            additional_params={
                "length": 260,
                "lanes": 1,
                "speed_limit": 30,
                "resolution": 40,
            }, ),
        veh=vehicles,
        initial=InitialConfig(),
    )
    return flow_params