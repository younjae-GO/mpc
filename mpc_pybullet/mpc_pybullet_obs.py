import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
from casadi_mpc import VehicleModel, MPC_soft, MPC_hard
from casadi_mpc.utils import compute_path_from_wp, get_ref_trajectory


# Params
TARGET_VEL = 1.0  # m/s
L = 0.3  # vehicle wheelbase [m]
T = 5  # Prediction Horizon [s]
DT = 0.2  # discretization step [s]

OBSTACLES = [
    {"pos": [4., -4., 0.2], "size": [0.15, 0.15, 0.4], "color": [1, 0, 0, 1]},
    {"pos": [4., 2., 0.2], "size": [0.2, 0.15, 0.4], "color": [0, 1, 0, 1]},
    {"pos": [6.3, 3.7, 0.2], "size": [0.15, 0.2, 0.4], "color": [0, 0, 1, 1]},
    {"pos": [10.3, 2.7, 0.2], "size": [0.2, 0.15, 0.4], "color": [1, 1, 0, 1]},
    {"pos": [12.3, -1.3, 0.2], "size": [0.15, 0.15, 0.4], "color": [1, 0, 1, 1]},
]


def create_obstacles():
    obstacle_ids = []
    for obs in OBSTACLES:
        collision_shape = p.createCollisionShape(
            p.GEOM_BOX, 
            halfExtents=obs["size"]
        )
        visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=obs["size"],
            rgbaColor=obs["color"]
        )
        obstacle_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=obs["pos"]
        )
        obstacle_ids.append(obstacle_id)
        p.addUserDebugLine(
            [obs["pos"][0] - obs["size"][0], obs["pos"][1] - obs["size"][1], 0],
            [obs["pos"][0] + obs["size"][0], obs["pos"][1] + obs["size"][1], obs["size"][2] * 2],
            lineColorRGB=[0.5, 0.5, 0.5],
            lineWidth=2
        )
    return obstacle_ids

def check_collision(car_id, obstacle_ids):
    for obs_id in obstacle_ids:
        contact_points = p.getContactPoints(bodyA=car_id, bodyB=obs_id)
        if len(contact_points) > 0:
            return True, obs_id
    return False, None

def get_state(robotId):
    robPos, robOrn = p.getBasePositionAndOrientation(robotId)
    linVel, angVel = p.getBaseVelocity(robotId)

    return np.array(
        [
            robPos[0],
            robPos[1],
            np.sqrt(linVel[0] ** 2 + linVel[1] ** 2),
            p.getEulerFromQuaternion(robOrn)[2],
        ]
    )

def set_ctrl(robotId, currVel, acceleration, steeringAngle):
    gearRatio = 1.0 / 21
    steering = [0, 2]
    wheels = [8, 15]
    maxForce = 50

    targetVelocity = currVel + acceleration * DT

    for wheel in wheels:
        p.setJointMotorControl2(
            robotId,
            wheel,
            p.VELOCITY_CONTROL,
            targetVelocity=targetVelocity / gearRatio,
            force=maxForce,
        )

    for steer in steering:
        p.setJointMotorControl2(
            robotId, steer, p.POSITION_CONTROL, targetPosition=steeringAngle
        )


def plot_results(path, x_history, y_history, obstacles_info=None):
    plt.style.use("ggplot")
    plt.figure(figsize=(12, 8))
    plt.title("MPC Tracking Results with Obstacles")
    plt.plot(
        path[0, :], path[1, :], c="tab:orange", marker=".", label="reference track"
    )
    plt.plot(
        x_history,
        y_history,
        c="tab:blue",
        marker=".",
        alpha=0.5,
        label="vehicle trajectory",
    )
    
    for obs in obstacles_info:
        pos = obs["pos"]
        size = obs["size"]
        color = obs["color"][:3]
        
        from matplotlib.patches import Rectangle
        rect = Rectangle(
            (pos[0] - size[0], pos[1] - size[1]), 
            2 * size[0], 
            2 * size[1],
            linewidth=2, 
            edgecolor='black', 
            facecolor=color,
            alpha=0.7
        )
        plt.gca().add_patch(rect)
    
    plt.axis("equal")
    plt.legend()
    plt.grid(True)
    plt.xlabel("X position (m)")
    plt.ylabel("Y position (m)")
    plt.show()


def run_sim():
    p.connect(p.GUI)
    p.resetDebugVisualizerCamera(
        cameraDistance=2.0,
        cameraYaw=-90,
        cameraPitch=-45,
        cameraTargetPosition=[5, 0, 0.65],
    )

    p.resetSimulation()

    p.setGravity(0, 0, -10)
    useRealTimeSim = 1

    p.setTimeStep(1.0 / 120.0)
    p.setRealTimeSimulation(useRealTimeSim)

    file_path = pathlib.Path(__file__).parent.resolve()
    plane = p.loadURDF(str(file_path) + "/racecar/plane.urdf")
    car = p.loadURDF(
        str(file_path) + "/racecar/f10_racecar/racecar_differential.urdf", [0, 0.3, 0.3]
    )

    obstacle_ids = create_obstacles()
    print(f"Created {len(obstacle_ids)} static obstacles")

    for wheel in range(p.getNumJoints(car)):
        p.setJointMotorControl2(
            car, wheel, p.VELOCITY_CONTROL, targetVelocity=0, force=0
        )

    constraints = [
        (9, 11, 1), (10, 13, -1), (9, 13, -1), (16, 18, 1), 
        (16, 19, -1), (17, 19, -1)
    ]
    
    for parent, child, ratio in constraints:
        c = p.createConstraint(
            car, parent, car, child,
            jointType=p.JOINT_GEAR,
            jointAxis=[0, 1, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
        p.changeConstraint(c, gearRatio=ratio, maxForce=10000)
    c = p.createConstraint(
        car, 1, car, 18,
        jointType=p.JOINT_GEAR,
        jointAxis=[0, 1, 0],
        parentFramePosition=[0, 0, 0],
        childFramePosition=[0, 0, 0],
    )
    p.changeConstraint(c, gearRatio=-1, gearAuxLink=15, maxForce=10000)
    c = p.createConstraint(
        car, 3, car, 19,
        jointType=p.JOINT_GEAR,
        jointAxis=[0, 1, 0],
        parentFramePosition=[0, 0, 0],
        childFramePosition=[0, 0, 0],
    )
    p.changeConstraint(c, gearRatio=-1, gearAuxLink=15, maxForce=10000)

    path = compute_path_from_wp(
        [0, 3, 4, 6, 10, 12, 12, 6, 1, 0],
        [0, 0, 2, 4, 3, -1, -1, -6, -2, -2],
        0.05,
    )
    for x_, y_ in zip(path[0, :], path[1, :]):
        p.addUserDebugLine([x_, y_, 0], [x_, y_, 0.33], [0, 0, 1])
    
    control = np.zeros(2)
    Q = [20, 20, 10, 20]
    Qf = [30, 30, 30, 30]
    R = [10, 10]
    P = [1, 1]

    #mpc = MPC_soft(VehicleModel(), T, DT, Q, Qf, R, P, OBSTACLES)
    mpc = MPC_hard(VehicleModel(), T, DT, Q, Qf, R, P, OBSTACLES)

    x_history = []
    y_history = []
    collision_count = 0
    simulation_time = 0
    

    while 1: 
        state = get_state(car)
        x_history.append(state[0])
        y_history.append(state[1])
        simulation_time += DT

        collision_static, obs_id = check_collision(car, obstacle_ids)

        if collision_static:
            collision_count += 1
            print(f"Collision detected with obstacle! Total collisions: {collision_count}")
            if collision_count > 5:
                print("Too many collisions! Stopping simulation.")
                break

        p.addUserDebugLine(
            [state[0], state[1], 0], [state[0], state[1], 0.5], [1, 0, 0]
        )

        if np.sqrt((state[0] - path[0, -1]) ** 2 + (state[1] - path[1, -1]) ** 2) < 0.3:
            set_ctrl(car, 0, 0, 0)
            plot_results(path, x_history, y_history, OBSTACLES)
            p.disconnect()
            return

        target = get_ref_trajectory(state, path, TARGET_VEL, T, DT)
        ego_state = np.array([0.0, 0.0, state[2], 0.0])

        ego_state[0] = ego_state[0] + ego_state[2] * np.cos(ego_state[3]) * DT
        ego_state[1] = ego_state[1] + ego_state[2] * np.sin(ego_state[3]) * DT
        ego_state[2] = ego_state[2] + control[0] * DT
        ego_state[3] = ego_state[3] + control[0] * np.tan(control[1]) / L * DT

        start = time.time()
        _, u_mpc = mpc.step(ego_state, target, control, state)
        control[0] = u_mpc[0, 0]
        control[1] = u_mpc[1, 0]
        elapsed = time.time() - start

        print(f"Time: {simulation_time:.1f}s, Optimization: {elapsed:.4f}s, Collisions: {collision_count}")

        set_ctrl(car, state[2], control[0], control[1])

        if DT - elapsed > 0:
            time.sleep(DT - elapsed)

    plot_results(path, x_history, y_history, OBSTACLES)
    p.disconnect()


if __name__ == "__main__":
    run_sim()