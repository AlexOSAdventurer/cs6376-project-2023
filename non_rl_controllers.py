"""Contains a list of custom velocity controllers."""

from flow.controllers.base_controller import BaseController
import numpy as np

class PISaturationWithDelay(BaseController):
    def __init__(self, veh_id, car_following_params):
        BaseController.__init__(self, veh_id, car_following_params, delay=0.0)

        # maximum achievable acceleration by the vehicle
        self.max_accel = car_following_params.controller_params['accel']

        # history used to determine AV desired velocity
        self.v_history = []

        # other parameters
        self.gamma = 2
        self.g_l = 7
        self.g_u = 30
        self.v_catch = 1

        # values that are updated by using their old information
        self.alpha = 0
        self.beta = 1 - 0.5 * self.alpha
        self.U = 0
        self.v_target = 0
        self.v_cmd = 0

    def get_accel(self, env):
        state = env.get_state()
        
        #lead_id = env.k.vehicle.get_leader(self.veh_id)
        #lead_vel = env.k.vehicle.get_speed(lead_id)
        #this_vel = env.k.vehicle.get_speed(self.veh_id)
        #dx = env.k.vehicle.get_headway(self.veh_id)
        this_vel = float(state[0]) * 15.
        lead_vel = (float(state[1]) * 15.) + this_vel
        dx = float(state[2]) * float(env.get_max_length()) 
        #print(this_vel, lead_vel, dx)

        dv = lead_vel - this_vel
        dx_s = max(2 * dv, 4)

        # update the AV's velocity history
        self.v_history.append(this_vel)

        if len(self.v_history) == int(38 / env.sim_step):
            del self.v_history[0]

        # update desired velocity values
        v_des = np.mean(self.v_history)
        v_target = v_des + self.v_catch \
            * min(max((dx - self.g_l) / (self.g_u - self.g_l), 0), 1)

        # update the alpha and beta values
        alpha = min(max((dx - dx_s) / self.gamma, 0), 1)
        beta = 1 - 0.5 * alpha

        # compute desired velocity
        self.v_cmd = beta * (alpha * v_target + (1 - alpha) * lead_vel) \
            + (1 - beta) * self.v_cmd

        # compute the acceleration
        accel = (self.v_cmd - this_vel) / env.sim_step

        return min(accel, self.max_accel)