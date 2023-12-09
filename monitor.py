

class Monitor:
    def __init__(self, max_length, max_speed):
        self.max_length = max_length
        self.max_speed = max_speed
        self.eps = 0.00001
    def evaluate_safety(self, vel_x, rel_vel, lead_dist):
        # Example safety evaluation logic
        vel_x = vel_x * self.max_speed
        rel_vel = rel_vel * self.max_speed
        lead_dist = lead_dist * self.max_length
        if vel_x > 6:
            return False  # Not safe if speed exceeds 6 m/s, which is above any of the potential optimal speeds for our rings
        elif lead_dist/(abs(vel_x) + self.eps) < 2:
            return False  # Not safe if time headway is less than 2 s
        elif rel_vel < 0:
            if lead_dist/(abs(rel_vel) + self.eps) < 2:
                return False  # Not safe if Time to Collision (TTC) is less than 2 s
            else:
                return True  # Safe otherwise
        else:
            return True  # Safe otherwise


'''
# Example usage
car_state = {
    'vel_x': 14,
    'rel_vel': 10,
    'lead_dist': 26
}

car_monitor = Monitor()

is_safe = car_monitor.evaluate_safety(car_state['vel_x'], car_state['rel_vel'], car_state['lead_dist'])

if is_safe:
    print("The car controller is safe.")
else:
    print("The car controller is not safe.")
'''