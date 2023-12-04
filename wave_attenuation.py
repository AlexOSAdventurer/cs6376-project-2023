import numpy as np
from flow.envs import WaveAttenuationPOEnv

class WaveAttenuationPOEnvWithDelay(WaveAttenuationPOEnv):
    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)
        print(self.sim_params.render, self.sim_params.save_render, self.should_render)
        self.observation = None
        self.observation_delay_time = env_params.additional_params['observation_delay']
        self.observation_delay_length = self.observation_delay_time / sim_params.sim_step
        self.previous_time_observation = 0

    def _apply_rl_actions(self, rl_actions):
        if rl_actions is None:
            return
        self.k.vehicle.apply_acceleration(
            self.k.vehicle.get_rl_ids(), rl_actions)

    def get_max_length(self):
        max_length = None
        if self.env_params.additional_params['ring_length'] is not None:
            max_length = self.env_params.additional_params['ring_length'][1]
        else:
            max_length = self.k.network.length()
        return max_length

    def update_state(self):
        if ((self.observation is not None) and ((self.time_counter - self.previous_time_observation) < self.observation_delay_length)):
            return
        #rl_id = self.k.vehicle.get_rl_ids()[0]
        #print(self.k.vehicle.get_ids())
        rl_id = "rl_0"
        lead_id = self.k.vehicle.get_leader(rl_id) or rl_id

        # normalizers
        max_speed = 15.
        max_length = self.get_max_length()

        self.observation = np.array([
            self.k.vehicle.get_speed(rl_id) / max_speed,
            (self.k.vehicle.get_speed(lead_id) -
             self.k.vehicle.get_speed(rl_id)) / max_speed,
            (self.k.vehicle.get_x_by_id(lead_id) -
             self.k.vehicle.get_x_by_id(rl_id)) % self.k.network.length()
            / max_length
        ])
        self.previous_time_observation = self.time_counter

    def get_state(self):
        self.update_state()
        return self.observation

    def additional_command(self):
        """Define which vehicles are observed for visualization purposes."""
        # specify observed vehicles
        rl_id = "rl_0"
        lead_id = self.k.vehicle.get_leader(rl_id) or rl_id
        self.k.vehicle.set_observed(lead_id)
        self.k.vehicle.set_color(rl_id, (255,0,0))