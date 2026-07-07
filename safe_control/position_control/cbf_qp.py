from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


class CBFQP:
    def __init__(self, robot, robot_spec, num_obs=1):
        self.robot = robot
        self.robot_spec = robot_spec
        self.num_obs = int(num_obs)
        self.model = robot_spec["model"]
        self.u_dim = 4 if self.model == "Quad3D" else 3 if self.model == "Manipulator2D" else 2
        self.cbf_param = {}

        if self.model in {"SingleIntegrator2D", "Unicycle2D", "Manipulator2D"}:
            self.cbf_param["alpha"] = 1.0
        elif self.model in {"DynamicUnicycle2D", "DoubleIntegrator2D", "KinematicBicycle2D", "Quad2D", "KinematicBicycle2D_C3BF", "KinematicBicycle2D_DPCBF", "Quad3D"}:
            self.cbf_param["alpha1"] = 1.5
            self.cbf_param["alpha2"] = 1.5

        if "cbf_alpha" in self.robot_spec:
            self.cbf_param["alpha"] = float(self.robot_spec["cbf_alpha"])
        if "cbf_alpha1" in self.robot_spec:
            self.cbf_param["alpha1"] = float(self.robot_spec["cbf_alpha1"])
        if "cbf_alpha2" in self.robot_spec:
            self.cbf_param["alpha2"] = float(self.robot_spec["cbf_alpha2"])

        self.setup_control_problem()

    def setup_control_problem(self):
        self.A1 = np.zeros((self.num_obs, self.u_dim), dtype=float)
        self.b1 = np.zeros((self.num_obs, 1), dtype=float)

    def _bounds(self):
        if self.model == "SingleIntegrator2D":
            lims = (-self.robot_spec["v_max"], self.robot_spec["v_max"])
            return [lims, lims]
        if self.model == "Unicycle2D":
            return [
                (-self.robot_spec["v_max"], self.robot_spec["v_max"]),
                (-self.robot_spec["w_max"], self.robot_spec["w_max"]),
            ]
        if self.model == "DynamicUnicycle2D":
            return [
                (-self.robot_spec["a_max"], self.robot_spec["a_max"]),
                (-self.robot_spec["w_max"], self.robot_spec["w_max"]),
            ]
        if self.model == "DoubleIntegrator2D":
            lims = (-self.robot_spec["a_max"], self.robot_spec["a_max"])
            return [lims, lims]
        if "KinematicBicycle2D" in self.model:
            return [
                (-self.robot_spec["a_max"], self.robot_spec["a_max"]),
                (-self.robot_spec["beta_max"], self.robot_spec["beta_max"]),
            ]
        if self.model == "Quad2D":
            return [
                (self.robot_spec["f_min"], self.robot_spec["f_max"]),
                (self.robot_spec["f_min"], self.robot_spec["f_max"]),
            ]
        if self.model == "Quad3D":
            u_max = self.robot_spec["u_max"]
            return [(0.0, u_max), (-u_max, u_max), (-u_max, u_max), (-u_max, u_max)]
        if self.model == "Manipulator2D":
            lims = (-self.robot_spec["w_max"], self.robot_spec["w_max"])
            return [lims, lims, lims]
        return [(-1.0, 1.0)] * self.u_dim

    def solve_control_problem(self, robot_state, control_ref, obs_list):
        u_ref = np.asarray(control_ref["u_ref"], dtype=float).reshape(-1)
        if obs_list is None:
            self.status = "optimal"
            return u_ref.reshape(-1, 1)

        self.A1.fill(0.0)
        self.b1.fill(0.0)

        mode = self.robot_spec.get("cbf_mode", "cbf")
        row_idx = 0
        for obs in obs_list:
            if obs is None or row_idx >= self.num_obs:
                continue

            if self.model == "Manipulator2D":
                h_list, dh_dx_list = self.robot.agent_barrier(obs)
                for h, dh_dx in zip(h_list, dh_dx_list):
                    if row_idx >= self.num_obs:
                        break
                    if mode == "hard":
                        dt = self.robot.dt
                        self.A1[row_idx, :] = (dh_dx @ self.robot.g()).reshape(-1)
                        self.b1[row_idx, 0] = float(h / dt + (dh_dx @ self.robot.f())[0, 0])
                    else:
                        self.A1[row_idx, :] = dh_dx
                        self.b1[row_idx, 0] = self.cbf_param["alpha"] * h
                    row_idx += 1
                continue

            dt = self.robot.dt
            if self.model in {"SingleIntegrator2D", "Unicycle2D", "KinematicBicycle2D_C3BF", "KinematicBicycle2D_DPCBF", "Quad3D"}:
                h, dh_dx = self.robot.agent_barrier(obs)
                if mode == "hard":
                    self.A1[row_idx, :] = (dh_dx @ self.robot.g()).reshape(-1)
                    self.b1[row_idx, 0] = float(h / dt + (dh_dx @ self.robot.f())[0, 0])
                else:
                    self.A1[row_idx, :] = (dh_dx @ self.robot.g()).reshape(-1)
                    self.b1[row_idx, 0] = float((dh_dx @ self.robot.f())[0, 0] + self.cbf_param["alpha"] * h)
            elif self.model in {"DynamicUnicycle2D", "DoubleIntegrator2D", "KinematicBicycle2D", "Quad2D"}:
                h, h_dot, dh_dot_dx = self.robot.agent_barrier(obs)
                if mode == "hard":
                    self.A1[row_idx, :] = (dh_dot_dx @ self.robot.g()).reshape(-1)
                    self.b1[row_idx, 0] = float(h / (dt**2) + 2 * h_dot / dt + (dh_dot_dx @ self.robot.f())[0, 0])
                else:
                    gamma1 = self.cbf_param["alpha1"] + self.cbf_param["alpha2"]
                    gamma2 = self.cbf_param["alpha1"] * self.cbf_param["alpha2"]
                    self.A1[row_idx, :] = (dh_dot_dx @ self.robot.g()).reshape(-1)
                    self.b1[row_idx, 0] = float((dh_dot_dx @ self.robot.f())[0, 0] + gamma1 * h_dot + gamma2 * h)
            row_idx += 1

        A = self.A1[:row_idx, :]
        b = self.b1[:row_idx, 0]
        bounds = self._bounds()
        x0 = np.clip(u_ref, [lo for lo, _ in bounds], [hi for _, hi in bounds])

        def objective(u):
            diff = u - u_ref
            return 0.5 * float(diff @ diff)

        def gradient(u):
            return u - u_ref

        constraints = []
        if A.size:
            constraints.append({"type": "ineq", "fun": lambda u, A=A, b=b: A @ u + b})

        res = minimize(
            objective,
            x0=x0,
            jac=gradient,
            bounds=bounds,
            constraints=constraints,
            method="SLSQP",
            options={"ftol": 1e-6, "maxiter": 50, "disp": False},
        )

        if not res.success or res.x is None:
            self.status = res.message if isinstance(res.message, str) else "failed"
            return u_ref.reshape(-1, 1)

        self.status = "optimal" if res.success else "failed"
        return np.asarray(res.x, dtype=float).reshape(-1, 1)
