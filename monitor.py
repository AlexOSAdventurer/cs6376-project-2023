class Monitor:
    def __init__(self, vel_x, rel_vel, lead_dist):
        self.vel_x = vel_x
        self.rel_vel = rel_vel
        self.lead_dist = lead_dist

    def evaluate_safety(self):
        # Example safety evaluation logic
        if self.vel_x > 15:
            return False  # Not safe if speed exceeds 15 m/s
        elif self.lead_dist/self.vel_x < 2:
            return False  # Not safe if time headway is less than 2 s
        elif self.rel_vel > 0:
            if self.lead_dist/self.vel_x < 2:
                return False  # Not safe if Time to Collision (TTC) is less than 2 s
            else:
                return True  # Safe otherwise
        else:
            return True  # Safe otherwise

# Example usage
car_state = {
    'vel_x': 14,
    'rel_vel': 10,
    'lead_dist': 26
}

car_monitor = Monitor(
    vel_x=car_state['vel_x'],
    rel_vel=car_state['rel_vel'],
    lead_dist=car_state['lead_dist']
)

is_safe = car_monitor.evaluate_safety()

if is_safe:
    print("The car controller is safe.")
else:
    print("The car controller is not safe.")