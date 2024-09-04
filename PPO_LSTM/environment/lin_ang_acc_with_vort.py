#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 15:17:08 2024

@author: federicatonti
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified FlowFieldNavEnv with Linear and Angular Accelerations in the State Space
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import h5py
import sys
from scipy.interpolate import RegularGridInterpolator
import ctypes
from collections import OrderedDict

class FlowFieldNavEnv(gym.Env):
    def __init__(self):
        super(FlowFieldNavEnv, self).__init__()
        
        file_path = '/Users/federicatonti/Desktop/KTH_work/2D_obstacles/files_to_use/combined_velocity_components_all.h5'
        self.flow_field_data = self.load_flow_field_data(file_path)
        
        X = self.flow_field_data['X']
        Y = self.flow_field_data['Y']
        U = self.flow_field_data['U']
        V = self.flow_field_data['V']
        
        
        max_x = np.max(X)
        min_x = np.min(X)
        max_y = np.max(Y)
        min_y = np.min(Y)
        
        max_vx = max(np.max(U), abs(np.min(U)))
        max_vy = max(np.max(V), abs(np.min(V)))
        
        self.domain_x_min = np.min(X)
        self.domain_x_max = np.max(X)
        self.domain_y_min = np.min(Y)
        self.domain_y_max = np.max(Y)
        
        np.random.seed(0)
                
        max_position = np.array([max_x, max_y])  
        max_velocity = np.array([max_vx, max_vy])  
        
        max_angle = np.pi  
        self.max_obstacle_distance = np.sqrt(max_x**2 + max_y**2) 
        max_dx = max_x - min_x  
        max_dy = max_y - min_y  
        
        self.sensor_angles = np.linspace(-np.pi, np.pi, 9) 
        self.sensor_ranges = np.zeros(len(self.sensor_angles)) 
        
        self.max_linear_acceleration = 1.0
        self.max_angular_acceleration = np.pi/4
        
        # Extend observation space to include linear and angular accelerations
        self.observation_space = spaces.Box(
            low=np.array([-np.pi, -np.pi, 0, -self.max_linear_acceleration, -self.max_angular_acceleration] + [0] * len(self.sensor_angles)),
            high=np.array([np.pi, np.pi, np.inf, self.max_linear_acceleration, self.max_angular_acceleration] + [np.inf] * len(self.sensor_angles)),
            dtype=np.float32
        )

        print("Initial observation shape:", self.observation_space.shape)
        
                
        self.action_space = gym.spaces.Box(
            low=np.array([-np.pi/4, -1.0]), 
            high=np.array([np.pi/4, 1.0]), 
            dtype=np.float32
        )

        self.obstacles = [
            {'x_range': [-0.25, 0.25], 'y_range': [0, 1]},  
            {'x_range': [1.25, 1.75], 'y_range': [0, 0.5]}  
        ]

        self.start_point = None
        self.plot = None
        self.heading_tolerance = np.pi/10
        self.larger_heading_tolerance = np.pi/2
        self.start_radius = 0.1
        self.target_radius = 0.2
        self.target_point = None
        self.plot = None
        self.state_history = []
        self.action_history = []
        self.observation_history = []
        self.reward_history = []
        self.done_history = []
        self.trajectory = []
        self.state = None
        self.observation = None
        self.current_time_step = 0
        self.next_time_step = 1
        self.progression_between_timesteps = 0.0
        self.max_timesteps = 80  
        self.max_episode_steps = 80  
        self.step_count = 0
        self.total_timesteps = 0  
        self.num_files = 300
        self.uav_state = np.zeros(8)  # Increased size to accommodate linear and angular accelerations
        self.uav_speed_in = 0.001 
        self.dt_simulation = 3.5e-5  
        self.steps_per_flow_field_timestep = 2500
        self.min_speed = -1.5
        self.max_speed = 1.5 
        self.max_angular_acceleration = np.pi/6 
        self.max_linear_acceleration = 0.3 
        self.gamma_eff = 0.3
        self.lambda_lin = 0.2
        self.lambda_ang = 0.5
        self.sigma = 8.0  
        self.alpha =3.0  
        self.beta = 25.0  
        self.r_free = 0.2  
        self.r_step = -0.2  
        self.dt = self.steps_per_flow_field_timestep * self.dt_simulation
        self.num_files = 300  
        
        self.total_simulation_duration = self.num_files * \
            (self.steps_per_flow_field_timestep * self.dt_simulation)
        
        
            # Load the C shared library
        # Load the C shared library and assign it to self.lib
        self.lib = ctypes.CDLL('./runge_kutta.so')
        
        # Define the argument types for the runge_kutta_4th function
        self.lib.runge_kutta_4th.argtypes = [
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), 
            ctypes.c_double, ctypes.c_double, 
            ctypes.c_double, ctypes.c_double, 
            ctypes.c_double, 
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), 
            ctypes.c_double, 
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), 
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), 
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), 
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)
        ]
        
        
        
    def normalize(self, value, min_value, max_value, range_min=-1.0, range_max=1.0):
        return (value - min_value) / (max_value - min_value) * (range_max - range_min) + range_min
    
    def denormalize(self, normalized_value, min_value, max_value):
        return normalized_value * (max_value - min_value) + min_value
    
    def normalize_observations(self, observations):
        normalized_observations = np.zeros_like(observations)
        for i, obs in enumerate(observations):
            if i == 0:  
                normalized_observations[i] = self.normalize(obs, -np.pi, np.pi, -1, 1)
            elif i == 1:  
                normalized_observations[i] = self.normalize(obs, -np.pi, np.pi, -1, 1)
            elif i == 2:  
                normalized_observations[i] = self.normalize(obs, 0, self.max_obstacle_distance, 0, 1)
            elif i in [3, 4, 5, 6]:  
                normalized_observations[i] = self.normalize(obs, -1.0, 1.0, -1, 1)
            else:  
                normalized_observations[i] = self.normalize(obs, 0, self.max_obstacle_distance, 0, 1)
        return normalized_observations
    
    def normalize_reward(self, reward):
        reward_min = -300  
        reward_max = 300  
        return self.normalize(reward, reward_min, reward_max, -1, 1)
    
    def normalize_trajectory(self, trajectory):
        return [self.normalize(position) for position in trajectory]

    def normalize_obstacle(self, obstacle):
        normalized_x_range = [self.normalize(x, self.domain_x_min, self.domain_x_max) for x in obstacle['x_range']]
        normalized_y_range = [self.normalize(y, self.domain_y_min, self.domain_y_max) for y in obstacle['y_range']]    
        return {'x_range': normalized_x_range, 'y_range': normalized_y_range}
    
    def normalize_domain(self):
        normalized_min_x = self.normalize(self.domain_x_min, self.domain_x_min, self.domain_x_max)
        normalized_max_x = self.normalize(self.domain_x_max, self.domain_x_min, self.domain_x_max)
        normalized_min_y = self.normalize(self.domain_y_min, self.domain_y_min, self.domain_y_max)
        normalized_max_y = self.normalize(self.domain_y_max, self.domain_y_min, self.domain_y_max)
        
        return (normalized_min_x, normalized_max_x), (normalized_min_y, normalized_max_y)

    def load_flow_field_data(self, h5_file_path):
        with h5py.File(h5_file_path, 'r') as file:
            X = np.array(file['X'])
            Y = np.array(file['Y'])
            U = np.array(file['U'])  
            V = np.array(file['V'])  
        return {'X': X, 'Y': Y, 'U': U, 'V': V}
    
    def calculate_initial_heading_angle(self):
        return np.random.uniform(0, np.pi/2)
    
    def create_interpolator_for_timestep(self, timestep):
        u_data_at_timestep = self.flow_field_data['U'][timestep]
        v_data_at_timestep = self.flow_field_data['V'][timestep]

        unique_x = self.flow_field_data['X'][:, 0]  
        unique_y = self.flow_field_data['Y'][0, :]  
        points = (unique_x, unique_y)

        u_interpolator = RegularGridInterpolator(points, u_data_at_timestep)
        v_interpolator = RegularGridInterpolator(points, v_data_at_timestep)

        return u_interpolator, v_interpolator
    
    def clamp_position(self, position):
        x_min, x_max = self.flow_field_data['X'].min(), self.flow_field_data['X'].max()
        y_min, y_max = self.flow_field_data['Y'].min(), self.flow_field_data['Y'].max()
        clamped_x = np.clip(position[0], x_min, x_max)      
        clamped_y = np.clip(position[1], y_min, y_max)
        return np.array([clamped_x, clamped_y])

    def interpolate_velocity(self, position, current_time_step, next_time_step, progression):
        max_time_step = len(self.flow_field_data['U'])
        next_time_step = (current_time_step + 1) % max_time_step  
        position = self.clamp_position(position)
        current_u_interpolator, current_v_interpolator = self.create_interpolator_for_timestep(current_time_step)
        next_u_interpolator, next_v_interpolator = self.create_interpolator_for_timestep(next_time_step)
        current_u = current_u_interpolator([position])[0]
        current_v = current_v_interpolator([position])[0]
        next_u = next_u_interpolator([position])[0]
        next_v = next_v_interpolator([position])[0]
        interpolated_u = (1 - progression) * current_u + progression * next_u
        interpolated_v = (1 - progression) * current_v + progression * next_v
        
        return np.array([interpolated_u, interpolated_v])

    def get_flow_field_velocity(self, position):
        clamped_position = self.clamp_position(position)
        next_time_step = (self.current_time_step + 1) % self.max_timesteps      
        return self.interpolate_velocity(clamped_position, self.current_time_step, next_time_step, self.progression_between_timesteps)
    
    def random_point_in_circle(self, center, radius):
        angle = np.random.uniform(0, 2 * np.pi)
        r = np.sqrt(np.random.uniform(0, 1)) * radius
        x = center[0] + r * np.cos(angle)
        y = center[1] + r * np.sin(angle)
        return np.array([x, y])
        
    def random_point_before_first_obstacle(self, min_x=-2, max_x=4, min_y=0, max_y=3):
        safe_margin = 0.1  
        start_x = np.random.uniform(-1.8, -1.0)
        start_y = np.random.uniform(0.2, 1.0)
        return np.array([start_x, start_y])

    def random_point_after_second_obstacle(self, min_x=-2, max_x=4, min_y=0, max_y=3):
        center_x = np.random.uniform(2.2, 3.8)
        center_y = np.random.uniform(0.2, 1.0)
        return np.array([center_x, center_y])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_time_step = 0
        self.step_count = 0
        self.total_timesteps = 0
        self.progression_between_timesteps = 0.0
        self.start_point = self.random_point_before_first_obstacle(min_x=-2, max_x=4, min_y=0, max_y=3)
        self.target_point = self.random_point_after_second_obstacle(min_x=-2, max_x=4, min_y=0, max_y=3)
        self.target_area_center = self.target_point
        initial_heading_angle = self.calculate_initial_heading_angle()
        initial_speed = self.uav_speed_in
        initial_velocity_x = initial_speed * np.cos(initial_heading_angle)
        initial_velocity_y = initial_speed * np.sin(initial_heading_angle)
        initial_angular_velocity = 0.0
        initial_linear_acceleration = 0.0
        initial_angular_acceleration = 0.0
        self.uav_state = np.array([
            *self.start_point,
            initial_heading_angle,
            initial_velocity_x, initial_velocity_y,
            initial_angular_velocity,
            initial_linear_acceleration,
            initial_angular_acceleration
        ])
        self.previous_velocity = np.array([initial_velocity_x, initial_velocity_y])
        self.state_history.clear()
        self.action_history.clear()
        self.observation_history.clear()
        self.reward_history.clear()
        self.trajectory.clear()
        self.uav_speed = self.get_flow_field_velocity(self.start_point)
        sensor_readings = self.update_sensor_readings(self.start_point, self.uav_state[2])
        observation = self.generate_observation(self.start_point, self.uav_state[2])
        self.state_history.append(self.uav_state.copy())
        self.observation_history.append(observation)
        self.reward_history.append(0)
        info = {}
        return observation, info
    
    def generate_observation(self, position, heading_angle):
        distance_to_target, angle_to_target = self.calculate_distance_and_angle_to_target(position)
        sensor_readings = self.update_sensor_readings(position, heading_angle)
        linear_acceleration = self.uav_state[6]
        angular_acceleration = self.uav_state[7]
        observation = np.concatenate(([heading_angle], [angle_to_target], [distance_to_target], 
                                      [linear_acceleration], [angular_acceleration], sensor_readings))
        return observation
    
    def transform_to_flow_relative(self, position, velocity, flow_velocity):
        relative_velocity = velocity - flow_velocity
        return position, relative_velocity

    def transform_back_to_stationary(self, position, relative_velocity, flow_velocity):
        absolute_velocity = relative_velocity + flow_velocity
        return position, absolute_velocity
    
    # def compute_vorticity(self, position, time):
    #     """
    #     Compute the vorticity at a given position and time using existing flow field velocity interpolation.
    #     Vorticity is calculated as the curl of the flow field: ω = ∂v/∂x - ∂u/∂y.
    #     """
    #     # Define a small offset for finite difference calculation
    #     epsilon = 1e-5
        
    #     # Flow velocity at the current position
    #     flow_velocity_now = self.get_flow_field_velocity(position)
        
    #     # Flow velocities at slightly offset positions
    #     flow_velocity_x_plus = self.get_flow_field_velocity([position[0] + epsilon, position[1]])
    #     flow_velocity_x_minus = self.get_flow_field_velocity([position[0] - epsilon, position[1]])
    #     flow_velocity_y_plus = self.get_flow_field_velocity([position[0], position[1] + epsilon])
    #     flow_velocity_y_minus = self.get_flow_field_velocity([position[0], position[1] - epsilon])
        
    #     # Compute partial derivatives using central differences
    #     partial_v_x = (flow_velocity_y_plus[1] - flow_velocity_y_minus[1]) / (2 * epsilon)  # ∂v/∂x
    #     partial_u_y = (flow_velocity_x_plus[0] - flow_velocity_x_minus[0]) / (2 * epsilon)  # ∂u/∂y
        
    #     # Vorticity is the difference between these partial derivatives
    #     vorticity = partial_v_x - partial_u_y
        
    #     return vorticity


    def runge_kutta_4th_ang(self, position, velocity, theta, omega, linear_acc, angular_acc, dt, time):
        # Prepare arguments for C function
        next_position = np.zeros(2)
        next_velocity = np.zeros(2)
        next_theta = ctypes.c_double()
        next_omega = ctypes.c_double()
    
        flow_velocity_now = self.get_flow_field_velocity(position)
        future_position = position + velocity * dt
        flow_velocity_next = self.get_flow_field_velocity(future_position)
    
        # Small offset for finite difference calculation
        epsilon = 1e-5
    
        # Flow velocities at slightly offset positions
        flow_velocity_x_plus = self.get_flow_field_velocity([position[0] + epsilon, position[1]])
        flow_velocity_x_minus = self.get_flow_field_velocity([position[0] - epsilon, position[1]])
        flow_velocity_y_plus = self.get_flow_field_velocity([position[0], position[1] + epsilon])
        flow_velocity_y_minus = self.get_flow_field_velocity([position[0], position[1] - epsilon])
    
        # Call the C function
        self.lib.runge_kutta_4th(
            position.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 
            velocity.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 
            ctypes.c_double(theta), ctypes.c_double(omega), 
            ctypes.c_double(linear_acc), ctypes.c_double(angular_acc), 
            ctypes.c_double(dt), 
            flow_velocity_now.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            flow_velocity_next.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 
            ctypes.c_double(epsilon), 
            flow_velocity_x_plus.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            flow_velocity_x_minus.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            flow_velocity_y_plus.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            flow_velocity_y_minus.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            next_position.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 
            next_velocity.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 
            ctypes.byref(next_theta), ctypes.byref(next_omega)
        )
    
        return next_position, next_velocity, next_theta.value, next_omega.value
    
    # def runge_kutta_4th_ang(self, position, velocity, theta, omega, linear_acc, angular_acc, dt, time):
    #     def dynamics(pos, vel, ang, ang_vel, lin_acc, ang_acc, time):
    #         # Get interpolated flow velocity using existing function
    #         flow_velocity_now = self.get_flow_field_velocity(pos)
            
    #         # Compute flow field acceleration by considering a small time step ahead
    #         future_position = pos + vel * dt  # Estimate future position
    #         flow_velocity_next = self.get_flow_field_velocity(future_position)
    #         flow_field_acceleration = (flow_velocity_next - flow_velocity_now) / dt
            
    #         # Compute vorticity using the existing flow field velocity function
    #         vorticity = self.compute_vorticity(pos, time)
            
    #         # Effective velocity considering the flow field
    #         effective_velocity = vel + flow_velocity_now
            
    #         # Dynamics equations
    #         dx_dt = effective_velocity[0]
    #         dy_dt = effective_velocity[1]
    #         dvx_dt = lin_acc * np.cos(ang) + flow_field_acceleration[0]
    #         dvy_dt = lin_acc * np.sin(ang) + flow_field_acceleration[1]
            
    #         # Include vorticity in the angular dynamics
    #         dtheta_dt = ang_vel
    #         domega_dt = ang_acc + vorticity  # Added vorticity effect
            
    #         return np.array([dx_dt, dy_dt]), np.array([dvx_dt, dvy_dt]), dtheta_dt, domega_dt
        
    #     # RK4 integration steps
    #     k1_pos, k1_vel, k1_theta, k1_omega = dynamics(position, velocity, theta, omega, linear_acc, angular_acc, time)
    #     k2_pos, k2_vel, k2_theta, k2_omega = dynamics(position + 0.5 * k1_pos * dt, velocity + 0.5 * k1_vel * dt, theta + 0.5 * k1_theta * dt, omega + 0.5 * k1_omega * dt, linear_acc, angular_acc, time + 0.5 * dt)
    #     k3_pos, k3_vel, k3_theta, k3_omega = dynamics(position + 0.5 * k2_pos * dt, velocity + 0.5 * k2_vel * dt, theta + 0.5 * k2_theta * dt, omega + 0.5 * k2_omega * dt, linear_acc, angular_acc, time + 0.5 * dt)
    #     k4_pos, k4_vel, k4_theta, k4_omega = dynamics(position + k3_pos * dt, velocity + k3_vel * dt, theta + k3_theta * dt, omega + k3_omega * dt, linear_acc, angular_acc, time + dt)
        
    #     # Update the state using RK4 weighted average
    #     next_position = position + (k1_pos + 2 * k2_pos + 2 * k3_pos + k4_pos) / 6 * dt
    #     next_velocity = velocity + (k1_vel + 2 * k2_vel + 2 * k3_vel + k4_vel) / 6 * dt
    #     next_theta = theta + (k1_theta + 2 * k2_theta + 2 * k3_theta + k4_theta) / 6 * dt
    #     next_omega = omega + (k1_omega + 2 * k2_omega + 2 * k3_omega + k4_omega) / 6 * dt
        
    #     # Clamp the velocity to max speed limits
    #     max_velocity = np.array([self.max_speed, self.max_speed])
    #     next_velocity = np.clip(next_velocity, -max_velocity, max_velocity)
        
    #     return next_position, next_velocity, next_theta, next_omega

    def step(self, action):
        angular_acceleration, linear_acceleration = action
    
        angular_acceleration = np.clip(angular_acceleration, -self.max_angular_acceleration, self.max_angular_acceleration)
        linear_acceleration = np.clip(linear_acceleration, -self.max_linear_acceleration, self.max_linear_acceleration)
        
        # Update UAV's state with the new accelerations
        self.uav_state[6] = linear_acceleration
        self.uav_state[7] = angular_acceleration
        
        # Store previous position
        self.prev_position = np.copy(self.uav_state[:2])
        
        # Current UAV state
        current_position = self.uav_state[:2]
        current_velocity = self.uav_state[3:5]
        new_speed = np.clip(current_velocity, self.min_speed, self.max_speed)
        current_heading = self.uav_state[2]
        current_angular_velocity = self.uav_state[5]
        
        # Smaller time step for sub-stepping within the RK4 method
        dt_substep = self.dt / 40
        
        # Perform integration with sub-steps
        for _ in range(40):
            # Call the C function for RK4 integration
            next_position, next_velocity, next_heading, next_angular_velocity = self.runge_kutta_4th_ang(
                current_position,
                current_velocity,
                current_heading,
                current_angular_velocity,
                linear_acceleration,
                angular_acceleration,
                dt_substep,
                self.current_time_step * self.dt  # Pass current simulation time to dynamics
            )
            
            # Normalize the heading angle within [-π, π]
            next_heading = np.mod(next_heading + np.pi, 2 * np.pi) - np.pi
            
            # Update the UAV state in the loop
            current_position = next_position
            current_velocity = next_velocity
            current_heading = next_heading
            current_angular_velocity = next_angular_velocity
        
        # Compute the new speed and adjust velocity vector
        new_speed = np.linalg.norm(current_velocity)
        current_velocity = new_speed * np.array([np.cos(current_heading), np.sin(current_heading)])
        
        # Update the UAV's state with final computed values
        self.uav_state[:2] = current_position
        self.uav_state[2] = current_heading
        self.uav_state[3:5] = current_velocity
        self.uav_state[5] = current_angular_velocity
        
        # Save the current position to the trajectory for tracking
        self.trajectory.append(current_position)
        
        # Calculate the reward and check if the episode is done
        reward, done = self.calculate_reward_and_done(self.uav_state, action)
        truncated = False
        
        # Check if the maximum number of steps in the episode has been reached
        if self.step_count >= self.max_episode_steps:
            truncated = True
            done = True
        
        # Generate the current observation based on the new UAV state
        observation = self.generate_observation(current_position, current_heading)
        
        # Update the progression between flow field timesteps
        self.progression_between_timesteps += 1
        if self.progression_between_timesteps >= 1.0:
            self.current_time_step = (self.current_time_step + 1) % len(self.flow_field_data['U'])
            self.progression_between_timesteps = 0
        
        # Update histories for logging purposes
        self.update_histories(action, observation, reward, done)
        
        # Increment the overall step count in the episode
        self.step_count += 1
        
        return observation, reward, done, truncated, {}



    # def step(self, action):
    #     angular_acceleration, linear_acceleration = action
        
    #     angular_acceleration = np.clip(angular_acceleration, -self.max_angular_acceleration, self.max_angular_acceleration)
    #     linear_acceleration = np.clip(linear_acceleration, -self.max_linear_acceleration, self.max_linear_acceleration)
        
    #     # Update UAV's state with the new accelerations
    #     self.uav_state[6] = linear_acceleration
    #     self.uav_state[7] = angular_acceleration
        
    #     # Store previous position
    #     self.prev_position = np.copy(self.uav_state[:2])
        
    #     # Current UAV state
    #     current_position = self.uav_state[:2]
    #     current_velocity = self.uav_state[3:5]
    #     current_heading = self.uav_state[2]
    #     current_angular_velocity = self.uav_state[5]
        
    #     # Smaller time step for sub-stepping within the RK4 method
    #     dt_substep = self.dt / 40
        
    #     # Perform integration with sub-steps
    #     for _ in range(40):
    #         next_position, next_velocity, next_heading, next_angular_velocity = self.runge_kutta_4th_ang(
    #             current_position,
    #             current_velocity,
    #             current_heading,
    #             current_angular_velocity,
    #             linear_acceleration,
    #             angular_acceleration,
    #             dt_substep,
    #             self.current_time_step * self.dt  # Pass current simulation time to dynamics
    #         )
            
    #         # Normalize the heading angle within [-π, π]
    #         next_heading = np.mod(next_heading + np.pi, 2 * np.pi) - np.pi
            
    #         # Update the UAV state in the loop
    #         current_position = next_position
    #         current_velocity = next_velocity
    #         current_heading = next_heading
    #         current_angular_velocity = next_angular_velocity
        
    #     # Compute the new speed and adjust velocity vector
    #     new_speed = np.linalg.norm(current_velocity)
    #     new_speed = np.clip(new_speed, self.min_speed, self.max_speed)
    #     current_velocity = new_speed * np.array([np.cos(current_heading), np.sin(current_heading)])
        
    #     # Update the UAV's state with final computed values
    #     self.uav_state[:2] = current_position
    #     self.uav_state[2] = current_heading
    #     self.uav_state[3:5] = current_velocity
    #     self.uav_state[5] = current_angular_velocity
        
    #     # Save the current position to the trajectory for tracking
    #     self.trajectory.append(current_position)
        
    #     # Calculate the reward and check if the episode is done
    #     reward, done = self.calculate_reward_and_done(self.uav_state, action)
    #     truncated = False
        
    #     # Check if the maximum number of steps in the episode has been reached
    #     if self.step_count >= self.max_episode_steps:
    #         truncated = True
    #         done = True
        
    #     # Generate the current observation based on the new UAV state
    #     observation = self.generate_observation(current_position, current_heading)
        
    #     # Update the progression between flow field timesteps
    #     self.progression_between_timesteps += 1
    #     if self.progression_between_timesteps >= 1.0:
    #         self.current_time_step = (self.current_time_step + 1) % len(self.flow_field_data['U'])
    #         self.progression_between_timesteps = 0
        
    #     # Update histories for logging purposes
    #     self.update_histories(action, observation, reward, done)
        
    #     # Increment the overall step count in the episode
    #     self.step_count += 1
        
    #     return observation, reward, done, truncated, {}

      
    def update_histories(self, action, observation, reward, done):
        self.action_history.append(action)
        self.observation_history.append(observation)
        self.reward_history.append(reward)
        self.done_history.append(done)
        self.state_history.append(self.uav_state.copy())
    
        history_limit = 10000  
        if len(self.action_history) > history_limit:
            self.action_history.pop(0)
            self.observation_history.pop(0)
            self.reward_history.pop(0)
            self.done_history.pop(0)
            self.state_history.pop(0)

    def check_collision(self, next_position):
        for obs in self.obstacles:
            if obs['x_range'][0] <= next_position[0] <= obs['x_range'][1] and \
               obs['y_range'][0] <= next_position[1] <= obs['y_range'][1]:
                return True
        return False
    
    def find_best_free_direction(self, current_position, current_heading, num_directions=9, look_ahead_distance=2.0):
        best_direction_relative = None
        max_free_distance = 0
    
        for i in range(num_directions):
            angle_relative = (2 * np.pi / num_directions) * i
            angle_absolute = current_heading + angle_relative  
    
            look_ahead_point = current_position + look_ahead_distance * np.array([np.cos(angle_absolute), np.sin(angle_absolute)])
            distance_to_obstacle = self.measure_distance_to_obstacle(current_position, angle_absolute)
    
            if distance_to_obstacle > max_free_distance:
                max_free_distance = distance_to_obstacle
                best_direction_relative = angle_relative  
        return best_direction_relative if best_direction_relative is not None else 0

    def update_sensor_readings(self, position, heading_angle):
        sensor_readings = []
        for angle in self.sensor_angles:
            heading_angle = np.clip(heading_angle, -np.pi, np.pi)
            absolute_angle = heading_angle + angle
            distance = self.measure_distance_to_obstacle(position, absolute_angle)
            sensor_readings.append(distance)
        return sensor_readings
    
    def calculate_distance_and_angle_to_target(self, position):
        vector_to_target = self.target_point - position
        distance_to_target = np.linalg.norm(vector_to_target)
        angle_to_target = np.arctan2(vector_to_target[1], vector_to_target[0])
        return distance_to_target, angle_to_target

    def measure_distance_to_obstacle(self, position, angle):
        min_distance = 100000000
        for obstacle in self.obstacles:
            distance = self.distance_from_point_to_rectangle(position, angle, obstacle)
            min_distance = min(min_distance, distance)
        return min_distance

    def distance_from_point_to_rectangle(self, point, angle, obstacle):
        edges = [
            ((obstacle['x_range'][0], obstacle['y_range'][0]), (obstacle['x_range'][0], obstacle['y_range'][1])),  
            ((obstacle['x_range'][1], obstacle['y_range'][0]), (obstacle['x_range'][1], obstacle['y_range'][1])),  
            ((obstacle['x_range'][0], obstacle['y_range'][0]), (obstacle['x_range'][1], obstacle['y_range'][0])),  
            ((obstacle['x_range'][0], obstacle['y_range'][1]), (obstacle['x_range'][1], obstacle['y_range'][1])),  
        ]
    
        distances = []
        for edge in edges:
            distance = self.calculate_intersection(point, angle, edge[0], edge[1])
            if distance is not None:
                distances.append(distance)
    
        if distances:
            return min(distances)
        else:
            return 100000000  

    def check_first_perspective_free_space(self, current_position):
        look_ahead_distance = 2.0  
        theta = self.uav_state[2]  
        look_ahead_point = np.array(current_position) + look_ahead_distance * np.array([np.cos(theta), np.sin(theta)])
    
        for obstacle in self.obstacles:
            x_min, x_max = obstacle['x_range']
            y_min, y_max = obstacle['y_range']
            x, y = look_ahead_point
    
            if x_min <= x <= x_max and y_min <= y <= y_max:
                return False  
        
        return True  

    def calculate_intersection(self, ray_origin, ray_angle, edge_start, edge_end):
        ray_direction = np.array([np.cos(ray_angle), np.sin(ray_angle)])
        start, end = np.array(edge_start[:2]), np.array(edge_end[:2])
    
        denom = ray_direction[0] * (start[1] - end[1]) - ray_direction[1] * (start[0] - end[0])
        if denom == 0:  
            return None
    
        t = ((start[0] - ray_origin[0]) * (start[1] - end[1]) - (start[1] - ray_origin[1]) * (start[0] - end[0])) / denom
        u = ((end[0] - start[0]) * (start[1] - ray_origin[1]) - (end[1] - start[1]) * (start[0] - ray_origin[0])) / denom
    
        if 0 <= u <= 1 and t > 0:  
            intersection_point = ray_origin + t * ray_direction
            return np.linalg.norm(intersection_point - ray_origin)  
        else:
            return None
        
    def calculate_reduced_distance(self, current_position, destination_position):
        prev_distance = np.linalg.norm(self.prev_position - destination_position)
        current_distance = np.linalg.norm(current_position - destination_position)
        d_dist = prev_distance - current_distance
        return d_dist

    def calculate_min_distance_to_obstacles(self, position, heading_angle):
        global_sensor_angles = heading_angle + self.sensor_angles
        distances = np.full(len(self.sensor_angles), np.inf)
        
        # Define the edges for each obstacle
        for obstacle in self.obstacles:
            edges = [
                (np.array([obstacle['x_range'][0], obstacle['y_range'][0]]), 
                 np.array([obstacle['x_range'][0], obstacle['y_range'][1]])),
                (np.array([obstacle['x_range'][1], obstacle['y_range'][0]]), 
                 np.array([obstacle['x_range'][1], obstacle['y_range'][1]])),
                (np.array([obstacle['x_range'][0], obstacle['y_range'][0]]), 
                 np.array([obstacle['x_range'][1], obstacle['y_range'][0]])),
                (np.array([obstacle['x_range'][0], obstacle['y_range'][1]]), 
                 np.array([obstacle['x_range'][1], obstacle['y_range'][1]]))
            ]
            
            # For each sensor angle, find the minimum distance to any obstacle edge
            for i, angle in enumerate(global_sensor_angles):
                ray_direction = np.array([np.cos(angle), np.sin(angle)])
                for edge_start, edge_end in edges:
                    distance = self.calculate_intersection(position, angle, edge_start, edge_end)
                    if distance is not None:
                        distances[i] = min(distances[i], distance)
                        
        return distances
    
    def is_out_of_bounds(self, position):
        return (
            position[0] <= self.domain_x_min or 
            position[0] >= self.domain_x_max or 
            position[1] <= self.domain_y_min or 
            position[1] >= self.domain_y_max
        )

    def calculate_reward_and_done(self, speculative_uav_state, action):
        reward = 0
        self.plot = False
        
        current_position = speculative_uav_state[:2]
        linear_acceleration, angular_acceleration = action
        
        d_dist = self.calculate_reduced_distance(current_position, self.target_point)
        
        d_min = self.calculate_min_distance_to_obstacles(current_position, speculative_uav_state[2])
        
        free_space = self.check_first_perspective_free_space(current_position)
        
        r_trans = self.sigma * d_dist
        
        average_distance = np.min(d_min)
        r_bar = -self.alpha * np.exp(-self.beta * average_distance)
        
        r_free = self.r_free if free_space else 0
        
        P_lin_accel = -self.lambda_lin * np.abs(linear_acceleration)
        P_ang_accel = -self.lambda_ang * np.abs(angular_acceleration)
        
        R_efficiency = self.gamma_eff / (1 + np.abs(linear_acceleration) + np.abs(angular_acceleration))
        
        # proximity_velocity_penalty = 0
        # distance_to_target = np.linalg.norm(current_position - self.target_point)
        # if distance_to_target <= 1.0:  # Define a range within which to reduce speed
        #     proximity_velocity_penalty = -0.2 * np.linalg.norm(speculative_uav_state[3:5])
        
        reward = r_trans + r_bar + r_free + R_efficiency + P_lin_accel + P_ang_accel + self.r_step #+ proximity_velocity_penalty
        
        flow_velocity = self.get_flow_field_velocity(current_position)
        propulsion_velocity = speculative_uav_state[3:5] - flow_velocity
        energy_penalty = -0.2 * np.linalg.norm(propulsion_velocity)
        
        reward += energy_penalty
        
        done = self.has_reached_target(current_position) or self.is_out_of_bounds(current_position) or self.check_collision(current_position)
        
        if done:
            if self.has_reached_target(current_position):
                self.plot = True
                reward += 50
                print("Target reached!")
            else:
                reward -= 20 if self.is_out_of_bounds(current_position) else 60 if self.check_collision(current_position) else 0
        
        return reward, done


    def has_reached_target(self, current_position):
        distance_to_target = np.linalg.norm(current_position - self.target_point)
        margin_of_error = 0.05  
        return abs(distance_to_target - self.target_radius) <= margin_of_error
