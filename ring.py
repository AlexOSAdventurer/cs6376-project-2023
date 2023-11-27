from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import VehicleParams, SumoCarFollowingParams
from flow.controllers import RLController, IDMController, ContinuousRouter
from wave_attenuation import WaveAttenuationPOEnvWithDelay
from flow.networks import RingNetwork

# time horizon of a single rollout
HORIZON = 6000
# number of rollouts per training iteration
N_ROLLOUTS = 20
# number of parallel workers
N_CPUS = 35

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
    acceleration_controller=(RLController, {}),
    routing_controller=(ContinuousRouter, {}),
    num_vehicles=1)

flow_params = dict(
    # name of the experiment
    exp_tag="stabilizing_the_ring",
    env_name=WaveAttenuationPOEnvWithDelay,
    network=RingNetwork,
    simulator='traci',
    sim=SumoParams(
        sim_step=0.1,
        render=False,
        save_render=False,
        restart_instance=False,
        #emission_path='simulation_data_delay_0'
    ),
    env=EnvParams(
        horizon=HORIZON,
        warmup_steps=750,
        clip_actions=False,
        additional_params={
            "max_accel": 1,
            "max_decel": 1,
            "ring_length": [220, 270],
            "observation_delay": 0.0,
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
