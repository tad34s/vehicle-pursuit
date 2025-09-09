import numpy as np
from scipy.optimize import minimize

from depth_net.projector import Projector


class ModelPredictiveControl:
    CAR_LEN = Projector.CAR_SIZES[2]
    FRICTION_DISCOUNT = 0.90  # Tune for steady speeds
    dt = 0.1
    max_accel_mps2 = 2.5  # Forward (m/s²)
    max_decel_mps2 = 5.0  # Braking/reverse (m/s²)
    desired_dist = 1.0  # Safe following distance (m); tune

    def __init__(
        self,
        weights={
            "x": 1,
            "y": 2,
            "theta": 1,
            "control": 0.1,
            "distance_cost": 0.5,
        },  # Higher y weight for distance priority
    ):
        self.horizon = 10
        self.weights = weights
        num_inputs = 2
        self.u = np.zeros(self.horizon * num_inputs)
        self.bounds = []
        max_steering_angle = 20
        self.max_in_rads = 20 * np.pi / 180
        for i in range(self.horizon):
            self.bounds += [[-1, 1]]
            self.bounds += [[-self.max_in_rads, self.max_in_rads]]

    def optimize_controls(self, current_speed_kph, leader_speed_kph, current_relative):
        """
        current_relative: [x_rel_m, y_rel_m, theta_rel_deg] (current relative position/heading)
        """
        self.u = np.zeros(self.horizon * 2)

        # self.u = np.delete(self.u, 0)
        # self.u = np.delete(self.u, 0)
        # self.u = np.append(self.u, self.u[-2])
        # self.u = np.append(self.u, self.u[-2])
        current_v_mps = current_speed_kph / 3.6
        v_lead_mps = leader_speed_kph / 3.6  # Leader speed for relative updates

        # Initial state: Current relative (x, y, theta negated for convention, v_follower)
        theta_rel_rad = (
            -current_relative[2] * np.pi / 180
        )  # Negate for +right -> - (ccw turn to correct)
        current_state = np.append(
            [current_relative[0], current_relative[1], theta_rel_rad], [current_v_mps]
        )

        # Fixed reference: [0, desired_dist, 0]
        reference = (0.0, self.desired_dist, 0.0)

        # Temp store for plant_model
        self.v_lead_mps = v_lead_mps

        u_solution = minimize(
            self.cost_function,
            self.u,
            args=(current_state, reference),
            method="SLSQP",
            bounds=self.bounds,
            tol=1e-3,
        )
        self.u = u_solution.x
        # print("curr relative state (m, rad, mps)", current_state)
        # print("curr v (kph)", current_speed_kph, "leader v (kph)", leader_speed_kph)
        # print("fixed ref", reference)
        # print("best_controls (pedal_norm, steering)", *self.u)
        next_state = self.plant_model(current_state, self.dt, self.u[0], self.u[1], v_lead_mps)
        # print("next rel state (m, rad, mps)", next_state)
        # print("next v (kph)", next_state[3] * 3.6)
        pedal_out = self.u[0]
        steering_out = -self.u[1]  # Negate steering for your convention
        # print("output actions", pedal_out, steering_out)
        return pedal_out, steering_out / self.max_in_rads

    def plant_model(self, prev_state, dt, pedal, steering, v_lead_mps):
        """Relative dynamics: Follower relative to fixed/moving leader (assumes leader straight, constant v_lead along y)."""
        x_t = prev_state[0]  # Rel lateral
        y_t = prev_state[1]  # Rel longitudinal (distance)
        theta_t = prev_state[2]  # Rel heading
        v_t = prev_state[3]  # Follower absolute speed

        # Relative kinematics (bicycle model approximation)
        omega = v_t * np.tan(steering) / self.CAR_LEN  # Follower turn rate (+ ccw/left)

        # dx/dt = v_f * sin(theta) - omega * y  (lateral: heading drift + rotation coupling)
        x_t += (v_t * np.sin(theta_t) - omega * y_t) * dt
        # dy/dt = v_lead - v_f * cos(theta) + omega * x  (long: leader advance - follower proj + rotation coupling)
        y_t += (v_lead_mps - v_t * np.cos(theta_t) + omega * x_t) * dt
        # dtheta/dt = -omega  (align heading: follower turns opposite to reduce rel theta)
        theta_t -= omega * dt
        theta_t = (theta_t + np.pi) % (2 * np.pi) - np.pi  # Wrap [-pi, pi]

        # Follower velocity (absolute; unchanged)
        if pedal >= 0:
            accel = pedal * self.max_accel_mps2
        else:
            accel = pedal * self.max_decel_mps2
        v_t = self.FRICTION_DISCOUNT * v_t + accel * dt
        v_t = np.clip(v_t, -12.5, 25.0)  # Realistic limits (m/s)

        return [x_t, y_t, theta_t, v_t]

    def cost_function(self, u, *args):
        current_state = args[0]
        reference = args[1]
        v_lead_mps = self.v_lead_mps  # From optimize_controls
        paths = []
        self.cost_x = 0
        self.cost_y = 0
        self.cost_theta = 0

        state = current_state
        ref = reference
        h = self.horizon
        for i in range(h):
            state = self.plant_model(state, self.dt, u[2 * i], u[2 * i + 1], v_lead_mps)
            paths.append([state[0], state[1]])  # Rel x/y for path length

            # Rel errors only (x, y to desired_dist, theta to 0)
            self.cost_x += (ref[0] - state[0]) ** 2
            self.cost_y += (ref[1] - state[1]) ** 2
            self.cost_theta += (ref[2] - state[2]) ** 2

        self.distance_cost = self.path_length(np.array(paths))

        # Average over horizon
        base_cost = (
            self.weights["x"] * (self.cost_x / h)
            + self.weights["y"] * (self.cost_y / h)
            + self.weights["theta"] * (self.cost_theta / h)
            + self.weights["distance_cost"] * self.distance_cost
        )

        # Control effort on pedal
        pedal_cost = np.sum(u[0::2] ** 2)
        total_cost = base_cost + self.weights["control"] * pedal_cost

        return total_cost

    def path_length(self, path):
        return np.sum(np.linalg.norm(path[1:] - path[:-1], axis=1))


if __name__ == "__main__":
    controller = ModelPredictiveControl()

    curr_state = [0.0, 0.0, 0.0, 26.334735870361328]
    ref = [5.249097, 15.89763, -1.615279639114708]
    best_controls = [
        0.9999999999985469,
        0.6733472084843167,
        1.0,
        0.01377048216188992,
        0.9999999999141727,
        0.03579596522901432,
        -0.9999999999999505,
        0.17340514732428233,
        -0.9999999999992842,
        0.4785261679252204,
        -0.9999999999995477,
        -0.7999999999996736,
        -1.0,
        -0.7999999999981018,
        -1.0,
        -0.7999999999976981,
        -0.9999999999731668,
        -0.799999999998049,
        0.0,
        -0.7999999999992503,
    ]

    cost_good = controller.cost_function(np.array(best_controls), curr_state, ref)
    print(cost_good)
    best_controls = [
        0.9999999999985469,
        0.6733472084843167,
        1.0,
        0.6733472084843167,
        0.9999999999141727,
        0.6733472084843167,
        0.9999999999999505,
        0.6733472084843167,
        0.9999999999992842,
        0.6733472084843167,
        0.9999999999995477,
        0.7999999999996736,
        1.0,
        0.3,
        1.0,
        0.2,
        0.9999999999731668,
        0.1,
        0.0,
        0.0,
    ]
    cost_bad = controller.cost_function(np.array(best_controls), curr_state, ref)
    print(cost_bad)
