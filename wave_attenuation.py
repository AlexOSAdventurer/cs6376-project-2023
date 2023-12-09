import numpy as np
from flow.envs import WaveAttenuationPOEnv
import monitor

class WaveAttenuationPOEnvWithDelay(WaveAttenuationPOEnv):
    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)
        print(self.sim_params.render, self.sim_params.save_render, self.should_render)
        self.observation = None
        self.observation_delay_time = env_params.additional_params['observation_delay']
        self.observation_delay_length = self.observation_delay_time / sim_params.sim_step
        self.previous_time_observation = 0
        self.max_speed = 15.
        self.monitor = monitor.Monitor(self.get_max_length(), self.max_speed)
        self.rl_id = "rl_0" # The cars by default follow a notation of carid_number, so the first car with id rl is rl_0

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

    def get_ground_truth_state(self):
        lead_id = self.k.vehicle.get_leader(self.rl_id) or self.rl_id

        # normalizers
        max_speed = self.max_speed
        max_length = self.get_max_length()

        observation = np.array([
            self.k.vehicle.get_speed(self.rl_id) / max_speed,
            (self.k.vehicle.get_speed(lead_id) -
             self.k.vehicle.get_speed(self.rl_id)) / max_speed,
            (self.k.vehicle.get_x_by_id(lead_id) -
             self.k.vehicle.get_x_by_id(self.rl_id)) % self.k.network.length()
            / max_length
        ])

        return observation

    def update_state(self):
        if ((self.observation is not None) and ((self.time_counter - self.previous_time_observation) < self.observation_delay_length)):
            return
        self.observation = self.get_ground_truth_state()
        self.previous_time_observation = self.time_counter

    def get_state(self):
        self.update_state()
        return self.observation

    def evaluate_safety(self):
        current_state = self.get_ground_truth_state()
        return (self.monitor.evaluate_safety(float(current_state[0]), float(current_state[1]), float(current_state[2])) and 1) or 0

    def additional_command(self):
        """Define which vehicles are observed for visualization purposes."""
        # specify observed vehicles
        lead_id = self.k.vehicle.get_leader(self.rl_id) or self.rl_id
        self.k.vehicle.set_observed(lead_id)
        self.k.vehicle.set_color(self.rl_id, (255,0,0))