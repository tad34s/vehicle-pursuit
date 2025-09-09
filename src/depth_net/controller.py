import numpy as np
from scipy.optimize import minimize

from depth_net.projector import Projector


class ModelPredictiveControl:
    CAR_LEN = Projector.CAR_SIZES[2]
    FRICTION_DISCOUNT = 0.95
    dt = 0.1

    def __init__(
        self,
        # start=[0, 0, 0],
        # reference=[10, 10, 0],
        weights={"x": 1, "y": 1, "theta": 1, "distance_cost": 0.5},
        # target_speed=0,
    ):
        self.horizon = 10

        # Reference or set point the controller will achieve.
        # self.reference = reference

        num_inputs = 2
        self.weights = weights
        self.u = np.zeros(self.horizon * num_inputs)
        self.bounds = []
        # self.current_state = np.append(start, [target_speed])

        # Set bounds for inputs bounded optimization.
        for i in range(self.horizon):
            self.bounds += [[-1, 1]]  # pedal
            self.bounds += [[-0.8, 0.8]]  # steering

    def optimize_controls(self, target_speed, reference):
        # self.u = np.delete(self.u, 0)
        # self.u = np.delete(self.u, 0)
        # self.u = np.append(self.u, self.u[-2])
        # self.u = np.append(self.u, self.u[-2])
        self.u = np.zeros(self.horizon * 2)
        current_state = np.append([0, 0, 0], [target_speed])
        reference = (reference[0], reference[1], -reference[2] * np.pi / 180)

        u_solution = minimize(
            self.cost_function,
            self.u,
            (current_state, reference),
            method="SLSQP",
            bounds=self.bounds,
            tol=1e-3,
        )
        # print("Step " + "   Time " + str(round(time.time() - start_time, 5)))
        self.u = u_solution.x
        print("curr state", *current_state)
        print("ref", *reference)
        print("best_controls", *self.u)
        print(
            "what it thinks it's the next state",
            *self.plant_model(current_state, self.dt, self.u[0], self.u[+1]),
        )

        # input("Press Enter to continue...")
        return self.u[0], -1 * self.u[1]

    def plant_model(self, prev_state, dt, pedal, steering):
        x_t = prev_state[0]
        y_t = prev_state[1]
        psi_t = prev_state[2]
        v_t = prev_state[3]

        x_t = x_t + v_t * np.sin(psi_t) * dt
        # y is forward, therefore should be cos (1 when angle 0)
        y_t = y_t + v_t * np.cos(psi_t) * dt
        psi_t = psi_t + v_t * dt * np.tan(steering) / self.CAR_LEN
        psi_t = (psi_t + np.pi) % (2 * np.pi) - np.pi
        v_t = self.FRICTION_DISCOUNT * v_t + pedal * dt

        return [x_t, y_t, psi_t, v_t]

    def cost_function(self, u, *args):
        current_state = args[0]
        reference = args[1]
        cost = 0.0
        paths = []
        self.cost_x = 0
        self.cost_y = 0
        self.cost_theta = 0

        state = current_state
        ref = reference
        for i in range(self.horizon):
            state = self.plant_model(state, self.dt, u[2 * i], u[2 * i + 1])
            paths.append([state[0], state[1]])

            self.cost_x += (ref[0] - state[0]) ** 2
            self.cost_y += (ref[1] - state[1]) ** 2
            self.cost_theta += (ref[2] - state[2]) ** 2
            # print(state)
            # print(
            #     self.cost_x,
            #     self.cost_y,
            #     self.cost_theta,
            # )

        # hausdorff = self.soft_hausdorff(np.array(paths), self.reference_path)
        self.distance_cost = self.path_length(np.array(paths))

        cost = (
            self.weights["x"] * self.cost_x
            + self.weights["y"] * self.cost_y
            + self.weights["theta"] * self.cost_theta
            + self.weights["distance_cost"] * self.distance_cost
        )

        # print(self.soft_hausdorff_flag)
        return cost

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
