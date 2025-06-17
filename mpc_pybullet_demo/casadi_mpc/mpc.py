import casadi as ca
import numpy as np


np.seterr(divide="ignore", invalid="ignore")


class VehicleModel:
    def __init__(self):
        self.wheelbase = 0.3
        self.max_speed = 1.5
        self.max_acc = 1.0
        self.max_steer = np.radians(30)


class MPC:
    def __init__(
        self,
        vehicle: VehicleModel,
        T: float,
        DT: float,
        state_cost: list,
        final_state_cost: list,
        input_cost: list,
        input_rate_cost: list,
    ):
        self.nx = 4  # number of state vars
        self.nu = 2  # number of input/control vars

        if len(state_cost) != self.nx:
            raise ValueError(f"State Error cost matrix shuld be of size {self.nx}")
        if len(final_state_cost) != self.nx:
            raise ValueError(f"End State Error cost matrix shuld be of size {self.nx}")
        if len(input_cost) != self.nu:
            raise ValueError(f"Control Effort cost matrix shuld be of size {self.nu}")
        if len(input_rate_cost) != self.nu:
            raise ValueError(
                f"Control Effort Difference cost matrix shuld be of size {self.nu}"
            )

        self.vehicle = vehicle
        self.dt = DT
        self.control_horizon = int(T / DT)
        self.Q = np.diag(state_cost)
        self.Qf = np.diag(final_state_cost)
        self.R = np.diag(input_cost)
        self.P = np.diag(input_rate_cost)

    def compute_linear_model_matrices(self, x_bar: list, u_bar: list):
        v = x_bar[2]
        theta = x_bar[3]

        a = u_bar[0]
        delta = u_bar[1]

        ct = np.cos(theta)
        st = np.sin(theta)
        cd = np.cos(delta)
        td = np.tan(delta)

        A = np.zeros((self.nx, self.nx))
        A[0, 2] = ct
        A[0, 3] = -v * st
        A[1, 2] = st
        A[1, 3] = v * ct
        A[3, 2] = v * td / self.vehicle.wheelbase
        A_lin = np.eye(self.nx) + self.dt * A

        B = np.zeros((self.nx, self.nu))
        B[2, 0] = 1
        B[3, 1] = v / (self.vehicle.wheelbase * cd**2)
        B_lin = self.dt * B

        f_xu = np.array([v * ct, v * st, a, v * td / self.vehicle.wheelbase]).reshape(
            self.nx, 1
        )
        C_lin = (
            self.dt
            * (
                f_xu
                - np.dot(A, x_bar.reshape(self.nx, 1))
                - np.dot(B, u_bar.reshape(self.nu, 1))
            ).flatten()
        )
        return A_lin, B_lin, C_lin

    def step(
        self,
        initial_state: list,
        target: list,
        prev_cmd: list,
    ):
        assert len(initial_state) == self.nx
        assert len(prev_cmd) == self.nu
        assert target.shape == (self.nx, self.control_horizon)

        x = ca.MX.sym('x', self.nx, self.control_horizon + 1)
        u = ca.MX.sym('u', self.nu, self.control_horizon)
        cost = 0
        g = []  # constraints

        A, B, C = self.compute_linear_model_matrices(initial_state, prev_cmd)

        # Tracking error cost
        for k in range(self.control_horizon):
            error = x[:, k + 1] - target[:, k]
            cost += ca.mtimes([error.T, self.Q, error])

        # Final point tracking cost
        final_error = x[:, -1] - target[:, -1]
        cost += ca.mtimes([final_error.T, self.Qf, final_error])

        # Actuation magnitude cost
        for k in range(self.control_horizon):
            cost += ca.mtimes([u[:, k].T, self.R, u[:, k]])

        # Actuation rate of change cost
        for k in range(1, self.control_horizon):
            du = u[:, k] - u[:, k - 1]
            cost += ca.mtimes([du.T, self.P, du])

        # Kinematics Constrains (equality constraints)
        for k in range(self.control_horizon):
            g.append(x[:, k + 1] - (A @ x[:, k] + B @ u[:, k] + C.flatten()))

        # initial state (equality constraint)
        g.append(x[:, 0] - initial_state)

        # Flatten optimization variables
        opt_variables = ca.vertcat(ca.reshape(x, -1, 1), ca.reshape(u, -1, 1))
        
        # Create NLP problem
        nlp = {
            'f': cost,
            'x': opt_variables,
            'g': ca.vertcat(*g)
        }

        # Solver options
        opts = {
            'ipopt.print_level': 0,
            'ipopt.sb': 'yes',
            'print_time': 0,
            'ipopt.max_iter': 100,
            'ipopt.warm_start_init_point': 'yes'
        }

        # Create solver
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        # Set up constraint bounds (equality constraints = 0)
        n_equality_constraints = (self.control_horizon + 1) * self.nx
        lbg = [0] * n_equality_constraints
        ubg = [0] * n_equality_constraints

        # Set up variable bounds (control input bounds)
        n_states = self.nx * (self.control_horizon + 1)
        n_controls = self.nu * self.control_horizon
        n_vars = n_states + n_controls
        
        # State bounds (no bounds on states)
        lbx = [-ca.inf] * n_states
        ubx = [ca.inf] * n_states
        
        # Control input bounds
        for k in range(self.control_horizon):
            # Acceleration bounds
            lbx.append(-self.vehicle.max_acc)
            ubx.append(self.vehicle.max_acc)
            # Steering bounds
            lbx.append(-self.vehicle.max_steer)
            ubx.append(self.vehicle.max_steer)

        # Initial guess (simple initialization)
        x0 = np.zeros(n_vars)

        try:
            sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
            
            x_opt = sol['x']
            
            x_sol = ca.reshape(x_opt[:n_states], self.nx, self.control_horizon + 1)
            u_sol = ca.reshape(x_opt[n_states:], self.nu, self.control_horizon)
            
            return np.array(x_sol), np.array(u_sol)
            
        except Exception as e:
            print(f"Optimization failed: {e}")
            x_fallback = np.tile(np.array(initial_state).reshape(-1, 1), (1, self.control_horizon + 1))
            u_fallback = np.tile(np.array(prev_cmd).reshape(-1, 1), (1, self.control_horizon))
            return x_fallback, u_fallback