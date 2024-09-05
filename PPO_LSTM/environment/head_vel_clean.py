#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 16:31:39 2024

@author: federicatonti
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 13:43:51 2024

@author: federicatonti
"""


import gymnasium as gym
from gymnasium import spaces
import numpy as np
import h5py
import sys
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import RectBivariateSpline
from joblib import Parallel, delayed
from interpolation import interpolate_velocity 
from rk4 import run_rk4

sys.path.append('/Users/federicatonti/Desktop/KTH_work/2D_obstacles/PPO/environment/')  # Adjust accordingly

class FlowFieldNavEnv(gym.Env):
    def __init__(self):
        super(FlowFieldNavEnv, self).__init__()

        
        file_path = '/Users/federicatonti/Desktop/KTH_work/2D_obstacles/files_to_use/combined_velocity_components_all.h5'
        self.flow_field_data = self.load_flow_field_data(file_path)
        # Access the loaded data
        X = self.flow_field_data['X']
        Y = self.flow_field_data['Y']
        U = self.flow_field_data['U']
        V = self.flow_field_data['V']
        

        
        max_x = np.max(X)
        min_x = np.min(X)
        max_y = np.max(Y)
        min_y = np.min(Y)
        
        # Determine the maximum velocity values (considering both positive and negative extremes)
        max_vx = max(np.max(U), abs(np.min(U)))
        max_vy = max(np.max(V), abs(np.min(V)))
        min_vx = min(np.max(U), abs(np.min(U)))
        min_vy = min(np.max(V), abs(np.min(V)))
        
        self.domain_x_min = np.min(X)
        self.domain_x_max = np.max(X)
        self.domain_y_min = np.min(Y)
        self.domain_y_max = np.max(Y)
        
        np.random.seed(0)
                
        max_position = np.array([max_x, max_y])  # Define these based on your environment
        max_velocity = np.array([max_vx, max_vy])  # Maximum velocity in each direction
        # Example: Generating file paths if they are sequentially named

        
        max_angle = np.pi  # Heading angle ranges from -pi to pi
       # Assuming max_x, max_y represent the maximum extent of your environment
        self.max_obstacle_distance = np.sqrt(max_x**2 + max_y**2) # Maximum observable distance to an obstacle
        max_dx = max_x - min_x  # The full width of the environment
        max_dy = max_y - min_y  # The full height of the environment
        max_distance = np.array([self.max_obstacle_distance])  # Max observable distance to an obstacle
        max_destination_distance = np.array([max_dx, max_dy])  # Max relative distance to destination
        
        self.sensor_angles = np.linspace(-np.pi, np.pi, 9) # Angles for range finders, in radians
        self.sensor_ranges = np.zeros(len(self.sensor_angles))  # Initialize sensor readings
        
        self.observation_space = spaces.Box(
        low=np.array([-np.pi, -np.pi, 0] + [0]*len(self.sensor_angles)),
        high=np.array([np.pi, np.pi, np.inf] + [np.inf]*len(self.sensor_angles)),
        dtype=np.float32
        )
        
        print(self.observation_space.shape[0])
                

        self.action_space = gym.spaces.Box(
            # [linear_acc_min, angular_acc_min]
            low=np.array([-np.pi, -2.0]),
            # [linear_acc_max, angular_acc_max]
            high=np.array([np.pi, 2.0]),
            dtype=np.float32
        )

        # # Load the flow field data
        # self.flow_field, self.x_range, self.y_range = self.load_flow_field_data(data_path)

        # Define obstacles
        self.obstacles = [
            {'x_range': [-0.25, 0.25], 'y_range': [0, 1]},  # Obstacle 1
            {'x_range': [1.25, 1.75], 'y_range': [0, 0.5]}  # Obstacle 2
        ]

        # Define the starting and target points
        self.start_point = None
        self.plot = False
        self.heading_tolerance = np.pi/10
        self.larger_heading_tolerance = np.pi/2
        self.start_radius = 0.1
        self.target_radius = 0.2
        self.target_point = None
        self.state_history = []
        self.action_history = []
        self.observation_history = []
        self.reward_history = []
        self.done_history = []
        self.trajectory = []
        # Initialize state
        self.state = None
        self.observation = None
        self.current_time_step = 0
        self.next_time_step = 1
        self.progression_between_timesteps = 0.0
        self.max_timesteps = 80  # Maximum number of timesteps per episode
        self.max_episode_steps = 80  # Maximum number of timesteps per episode
        self.step_count = 0
        self.total_timesteps = 0  # Total timesteps taken in the current episode
        self.num_files = 300
        self.uav_state = np.zeros(6)
        self.uav_speed_in = 0.0  # 1.7979089498488896# Set the initial UAV speed
        # self.uav_max_acceleration = 1.0
        # Time step details
        self.dt_simulation = 3.5e-5  # Duration of each simulation step
        # Number of simulation steps per flow field timestep
        self.steps_per_flow_field_timestep = 2500
        self.min_speed = -0.2
        self.max_speed = 2.0
        self.sigma = 8.0  # Positive constant for the transition reward
        self.alpha =3.0  # Positive constant for the obstacle penalty
        self.beta = 25.0  # Positive constant for the obstacle penalty exponent
        self.r_free = 0.2  # Constant free-space reward
        self.r_step = -0.6  # Constant step penalty
        # Duration of each simulation step in seconds
        self.dt = self.steps_per_flow_field_timestep * self.dt_simulation
        # self.dt = 0.04
        self.num_files = 300  # Total number of flow field files (snapshots)
        # Calculate the total simulation duration
        self.total_simulation_duration = self.num_files * \
            (self.steps_per_flow_field_timestep * self.dt_simulation)
        
    def normalize(self, value, min_value, max_value, range_min=-1.0, range_max=1.0):
        """Normalize a value from its actual range to a target range [range_min, range_max]."""
        return (value - min_value) / (max_value - min_value) * (range_max - range_min) + range_min
    
    def denormalize(self, normalized_value, min_value, max_value):
        return normalized_value * (max_value - min_value) + min_value
    
    def normalize_observations(self, observations):
        normalized_observations = np.zeros_like(observations)
        # Normalize each component of the observations
        for i, obs in enumerate(observations):
            if i == 0:  # Heading angle
                normalized_observations[i] = self.normalize(obs, -np.pi, np.pi, -1, 1)
            elif i == 1:  # Angle to target
                normalized_observations[i] = self.normalize(obs, -np.pi, np.pi, -1, 1)
            elif i == 2:  # Distance to target
                normalized_observations[i] = self.normalize(obs, 0, self.max_obstacle_distance, 0, 1)
            else:  # Sensor readings
                normalized_observations[i] = self.normalize(obs, 0, self.max_obstacle_distance, 0, 1)
        return normalized_observations
    
    def normalize_reward(self, reward):
        # Define these values based on the observed or expected rewards
        reward_min = -300  # For example, a collision penalty
        reward_max = 300  # For example, reaching the target reward
        return self.normalize(reward, reward_min, reward_max, -1, 1)
    
    def normalize_trajectory(self, trajectory):
        return [self.normalize(position) for position in trajectory]

    def normalize_obstacle(self, obstacle):
        # Assuming obstacle is a dictionary with 'x_range' and 'y_range'
        normalized_x_range = [self.normalize(x, self.domain_x_min, self.domain_x_max) for x in obstacle['x_range']]
        normalized_y_range = [self.normalize(y, self.domain_y_min, self.domain_y_max) for y in obstacle['y_range']]    
        return {'x_range': normalized_x_range, 'y_range': normalized_y_range}
    
    def normalize_domain(self):
    # Normalize the corners of the domain
        normalized_min_x = self.normalize(self.domain_x_min, self.domain_x_min, self.domain_x_max)
        normalized_max_x = self.normalize(self.domain_x_max, self.domain_x_min, self.domain_x_max)
        normalized_min_y = self.normalize(self.domain_y_min, self.domain_y_min, self.domain_y_max)
        normalized_max_y = self.normalize(self.domain_y_max, self.domain_y_min, self.domain_y_max)
        
        return (normalized_min_x, normalized_max_x), (normalized_min_y, normalized_max_y)


    def load_flow_field_data(self, h5_file_path):
        with h5py.File(h5_file_path, 'r') as file:
            X = np.array(file['X'])
            Y = np.array(file['Y'])
            U = np.array(file['U'])  # Assuming shape (time_steps, x_points, y_points)
            V = np.array(file['V'])  # Assuming shape (time_steps, x_points, y_points)
        return {'X': X, 'Y': Y, 'U': U, 'V': V}
    
    
    
    def calculate_initial_heading_angle(self):
        return np.random.uniform(0, np.pi/2)

    
    def clamp_position(self, position):
        # Implement position clamping logic to ensure it's within bounds
        clamped_x = np.clip(position[0], np.min(self.flow_field_data['X']), np.max(self.flow_field_data['X']))
        clamped_y = np.clip(position[1], np.min(self.flow_field_data['Y']), np.max(self.flow_field_data['Y']))
        return np.array([clamped_x, clamped_y])


    def get_flow_field_velocity(self, position):
        clamped_position = self.clamp_position(position)
    
        # Ensure you extract all required timesteps (300 in this case)
        u_field = self.flow_field_data['U']  # This will give you the full 300 timesteps
        v_field = self.flow_field_data['V']  # The full 300 timesteps
    
        x_grid = self.flow_field_data['X']
        y_grid = self.flow_field_data['Y']
        
        # Print to verify correct shapes
        # print(f"x_grid shape: {x_grid.shape}")
        # print(f"y_grid shape: {y_grid.shape}")
        # print(f"u_field shape: {u_field.shape}")
        # print(f"v_field shape: {v_field.shape}")
        
        # Extract the progression between timesteps
        progression = self.progression_between_timesteps
        
        # Call the Cython interpolation function with the correct u_field and v_field
        interpolated_velocity = interpolate_velocity(clamped_position, x_grid, y_grid, u_field, v_field, progression)
        
        return interpolated_velocity


    
    def random_point_in_circle(self, center, radius):
        angle = np.random.uniform(0, 2 * np.pi)
        r = np.sqrt(np.random.uniform(0, 1)) * radius
        x = center[0] + r * np.cos(angle)
        y = center[1] + r * np.sin(angle)
        return np.array([x, y])
        
    def random_point_before_first_obstacle(self, min_x=-2, max_x=4, min_y=0, max_y=3):
        # Ensure the starting area is well within the domain boundaries
        safe_margin = 0.1  # Define a safe margin distance to ensure start point is away from the boundary
        start_x_min = min_x + safe_margin
        start_x_max = self.obstacles[0]['x_range'][0] - self.start_radius - safe_margin
        start_y_min = min_y + safe_margin
        start_y_max = 1.0 - safe_margin
    
        # Generate a random start point within the defined range
        start_x = np.random.uniform(-1.8, -1.0)
        start_y = np.random.uniform(0.2, 1.0)
        # # Define two separate ranges for start_x
        # range_1 = (-1.9, -0.8)
        # range_2 = (0.4, 1.0)
    
        # Randomly choose between the two ranges
        # if np.random.rand() < 0.5:
        #     start_x = np.random.uniform(*range_1)
        # else:
        #     start_x = np.random.uniform(*range_2)
        # start_x = -1.5
        # start_y = 0.5
        
        return np.array([start_x, start_y])


    def random_point_after_second_obstacle(self, min_x=-2, max_x=4, min_y=0, max_y=3):
        """
        Define a center for the target area after the second obstacle within the domain boundaries.
        
        Parameters:
        - min_x: Minimum x-coordinate of the domain.
        - max_x: Maximum x-coordinate of the domain.
        - min_y: Minimum y-coordinate of the domain.
        - max_y: Maximum y-coordinate of the domain.
        """
        # Calculate the x-coordinate range after the second obstacle
        # start_x_after_second_obstacle = self.obstacles[1]['x_range'][1] + self.target_area_radius + 0.1
    
        # # Ensure the starting x-coordinate is within the domain boundaries
        # start_x = np.clip(start_x_after_second_obstacle, min_x, max_x)
    
        # Randomly generate a center point within the restricted domain
        center_x = np.random.uniform(2.2, 3.8)
        center_y = np.random.uniform(0.2, 1.0)
        # center_x = 3.5
        # center_y = 0.5
        
        # center = np.array([center_x, center_y])
        
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
        self.uav_state = np.array([
            *self.start_point,
            initial_heading_angle,
            initial_velocity_x, initial_velocity_y,
            initial_angular_velocity
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
        # Based on the UAV's current state, generate the observation for the agent
        distance_to_target, angle_to_target = self.calculate_distance_and_angle_to_target(position)
        sensor_readings = self.update_sensor_readings(position, heading_angle)
        # Observation could include heading angle, angle to target, distance to target, and sensor readings
        observation = np.concatenate(([heading_angle], [angle_to_target], [distance_to_target], sensor_readings))
        # print("Observation: heading_angle:", [heading_angle], "angle_to_target:", [angle_to_target], "distance_to_target:", [distance_to_target], "sensor_readings:", sensor_readings )
        return observation

    
    def get_uav_current_position(self):
    # Assuming the first two elements of self.uav_state are the x and y positions
        return self.uav_state[:2]
    
    def get_uav_current_heading_angle(self):
    # Assuming the first two elements of self.uav_state are the x and y positions
        return self.uav_state[2]
    
    def calculate_uav_velocity_with_flow_step(self, flow_velocity, heading_angle, speed):
        """
        Calculate UAV's velocity vector considering the flow field.
        """
        u_flow, v_flow = flow_velocity
        u_uav = speed * np.cos(heading_angle) + u_flow
        v_uav = speed * np.sin(heading_angle) + v_flow
        abs_vel = np.sqrt(u_uav**2 + v_uav**2)
        return u_uav, v_uav


    def step(self, action):
        desired_heading, desired_speed = action
        self.prev_position = np.copy(self.uav_state[:2])
        current_position = np.array(self.uav_state[:2], dtype=np.float64)  # Ensure it's a NumPy array
        current_velocity = np.array(self.uav_state[3:5], dtype=np.float64)  # Ensure it's a NumPy array
        current_heading = self.uav_state[2]
        current_angular_velocity = self.uav_state[5]
    
        dt_substep = self.dt / 40 # Example: using smaller sub-steps
    
        # Call your Cython RK4 method instead of the Python implementation
        for _ in range(40):
            u_flow, v_flow = self.get_flow_field_velocity(current_position)  # Calculate flow field velocity
            # print (u_flow,v_flow)
    
            # Call the Cython RK4 integration
            next_position, next_velocity, next_heading, next_angular_velocity = run_rk4(
                current_position, current_velocity, current_heading, current_angular_velocity, 
                desired_speed, desired_heading, dt_substep, u_flow, v_flow
            )
    
            # Update current state
            next_heading = np.mod(next_heading + np.pi, 2 * np.pi) - np.pi  # Keep heading in valid range
            current_position = np.array(next_position, dtype=np.float64)  # Ensure arrays for next step
            current_velocity = np.array(next_velocity, dtype=np.float64)
    
            current_heading = next_heading
            current_angular_velocity = next_angular_velocity
    
        # Update UAV state
        new_speed = np.linalg.norm(current_velocity)
        current_velocity = new_speed * np.array([np.cos(current_heading), np.sin(current_heading)])
    
        self.uav_state[:2] = current_position
        self.trajectory.append(current_position)
        self.uav_state[2] = current_heading
        self.uav_state[3:5] = current_velocity
        self.uav_state[5] = current_angular_velocity
    
        # Calculate reward and check for termination
        reward, done = self.calculate_reward_and_done(self.uav_state)
        truncated = False
        if self.step_count >= self.max_episode_steps:
            truncated = True
            done = True
    
        observation = self.generate_observation(current_position, current_heading)
    
        # Handle progression between time steps
        self.progression_between_timesteps += dt_substep
        if self.progression_between_timesteps >= dt_substep:
            self.current_time_step = (self.current_time_step + 1) % len(self.flow_field_data['U'])
            self.progression_between_timesteps = 0
    
        self.update_histories(action, observation, reward, done)
        self.step_count += 1
    
        return observation, reward, done, truncated, {}


      
    def update_histories(self, action, observation, reward, done):
    # Assuming histories are lists initialized in the __init__ method
        self.action_history.append(action)
        self.observation_history.append(observation)
        self.reward_history.append(reward)
        self.done_history.append(done)
        # You might also want to record the entire state or specific state components
        self.state_history.append(self.uav_state.copy())
    
        # Optional: Limit the history size to keep memory usage in check
        # This is especially important if you're not using these histories directly in a replay buffer
        # but still want to keep recent history for analysis or other purposes.
        history_limit = 10000  # Example limit
        if len(self.action_history) > history_limit:
            self.action_history.pop(0)
            self.observation_history.pop(0)
            self.reward_history.pop(0)
            self.done_history.pop(0)
            self.state_history.pop(0)


    def check_collision(self, next_position):
        # Check if the next_position collides with any obstacle
        # This is internal logic and does not expose the full state to the agent
        for obs in self.obstacles:
            if obs['x_range'][0] <= next_position[0] <= obs['x_range'][1] and \
               obs['y_range'][0] <= next_position[1] <= obs['y_range'][1]:
                return True
        return False
    
    def find_best_free_direction(self, current_position, current_heading, num_directions=9, look_ahead_distance=1.0):
        best_direction_relative = None
        max_free_distance = 0
    
        for i in range(num_directions):
            # Calculate the angle to check relative to the UAV's current heading
            angle_relative = (2 * np.pi / num_directions) * i
            angle_absolute = current_heading + angle_relative  # Convert to absolute angle
    
            # Calculate the look-ahead point in this direction
            look_ahead_point = current_position + look_ahead_distance * np.array([np.cos(angle_absolute), np.sin(angle_absolute)])
            # Measure the distance to the nearest obstacle in this direction
            distance_to_obstacle = self.measure_distance_to_obstacle(current_position, angle_absolute)
    
            if distance_to_obstacle > max_free_distance:
                max_free_distance = distance_to_obstacle
                best_direction_relative = angle_relative  # Store as relative angle
        # print("Best_direction_relative:", best_direction_relative)
    
        # Return the relative best direction or current heading if no better option
        return best_direction_relative if best_direction_relative is not None else 0



    def update_sensor_readings(self, position, heading_angle):
        sensor_readings = []
        for angle in self.sensor_angles:
            # Adjust sensor angle based on UAV's heading angle
            # print("Heading angle before clipping:", heading_angle)
            heading_angle = np.clip(heading_angle, -np.pi, np.pi)
            absolute_angle = heading_angle + angle
            # print("Absolute_angle: heading_angle:", heading_angle, "+ angle:", angle)
            distance = self.measure_distance_to_obstacle(position, absolute_angle)
            # print("Distance in update sensor readings:", distance)
            sensor_readings.append(distance)
        return sensor_readings
    
    import numpy as np
    

    def calculate_distance_and_angle_to_target(self, position):
        """
        Calculate the distance and angle from the UAV's current position to the target.
    
        Parameters:
        - position: The current position of the UAV as a numpy array [x, y].
    
        Returns:
        - distance_to_target: The Euclidean distance from the UAV to the target.
        - angle_to_target: The angle from the UAV's current position to the target, relative to the global frame.
        """
        # Vector from UAV's current position to the target
        vector_to_target = self.target_point - position
    
        # Calculate distance using Euclidean norm
        distance_to_target = np.linalg.norm(vector_to_target)
        # print("Distance_to_target:", distance_to_target)
    
        # Calculate angle using arctan2 (returns angle in radians between [-pi, pi])
        angle_to_target = np.arctan2(vector_to_target[1], vector_to_target[0])
        # print ("Angle to target:", angle_to_target)
    
        return distance_to_target, angle_to_target



    def measure_distance_to_obstacle(self, position, angle):
        # Measure the distance from the UAV's position in the direction given by angle to the nearest obstacle
        min_distance = 100000000
        for obstacle in self.obstacles:
            distance = self.distance_from_point_to_rectangle(position, angle, obstacle)
            min_distance = min(min_distance, distance)
        return min_distance


    def distance_from_point_to_rectangle(self, point, angle, obstacle):
        # Define edges based on the obstacle's x_range and y_range
        edges = [
            ((obstacle['x_range'][0], obstacle['y_range'][0]), (obstacle['x_range'][0], obstacle['y_range'][1])),  # Left edge
            ((obstacle['x_range'][1], obstacle['y_range'][0]), (obstacle['x_range'][1], obstacle['y_range'][1])),  # Right edge
            ((obstacle['x_range'][0], obstacle['y_range'][0]), (obstacle['x_range'][1], obstacle['y_range'][0])),  # Bottom edge
            ((obstacle['x_range'][0], obstacle['y_range'][1]), (obstacle['x_range'][1], obstacle['y_range'][1])),  # Top edge
        ]
    
        distances = []
        for edge in edges:
            # Calculate intersection with edge
            distance = self.calculate_intersection(point, angle, edge[0], edge[1])
            # print ("Distance from calculate intersection in distance from point to rectangle:", distance)
            if distance is not None:
                distances.append(distance)
    
        if distances:
            return min(distances)
        else:
            return 100000000  # No intersection if no distances are found

    def check_first_perspective_free_space(self, current_position):
        look_ahead_distance = 1.0  # Distance to look ahead from the current position
        theta = self.uav_state[2]  # UAV's current heading angle
        look_ahead_point = np.array(current_position) + look_ahead_distance * np.array([np.cos(theta), np.sin(theta)])
    
        for obstacle in self.obstacles:
            x_min, x_max = obstacle['x_range']
            y_min, y_max = obstacle['y_range']
            x, y = look_ahead_point
    
            # Check if the look-ahead point is within any obstacle range
            if x_min <= x <= x_max and y_min <= y <= y_max:
                return False  # First-perspective direction points to a place with an obstacle
        
        return True  # First-perspective direction is free of obstacles


    def calculate_intersection(self, ray_origin, ray_angle, edge_start, edge_end):
        # Calculate the direction vector of the ray based on the angle
        ray_direction = np.array([np.cos(ray_angle), np.sin(ray_angle)])
        start, end = np.array(edge_start[:2]), np.array(edge_end[:2])
    
        # Calculate the denominator to check for parallel lines
        denom = ray_direction[0] * (start[1] - end[1]) - ray_direction[1] * (start[0] - end[0])
        if denom == 0:  # Lines are parallel
            return None
    
        # Calculate the intersection point
        t = ((start[0] - ray_origin[0]) * (start[1] - end[1]) - (start[1] - ray_origin[1]) * (start[0] - end[0])) / denom
        u = ((end[0] - start[0]) * (start[1] - ray_origin[1]) - (end[1] - start[1]) * (start[0] - ray_origin[0])) / denom
    
        if 0 <= u <= 1 and t > 0:  # Intersection is within the segment and in the ray's direction
            intersection_point = ray_origin + t * ray_direction
            return np.linalg.norm(intersection_point - ray_origin)  # Return distance to intersection
        else:
            return None
        
    def calculate_reduced_distance(self, current_position, destination_position):
        """
        Calculates the reduction in distance towards the destination position from the previous timestep to the current position.
    
        Parameters:
        - current_position: The current position of the UAV (numpy array of shape [2]).
        - destination_position: The position of the destination (numpy array of shape [2]).
    
        Returns:
        - The change in distance towards the destination. A positive value indicates that the UAV has moved closer to the destination.
        """
        # Calculate the Euclidean distance from the previous position to the destination
        prev_distance = np.linalg.norm(self.prev_position - destination_position)
        # Calculate the Euclidean distance from the current position to the destination
        current_distance = np.linalg.norm(current_position - destination_position)
        # Calculate the reduction in distance (positive value indicates moving closer)
        d_dist = prev_distance - current_distance
        return d_dist
    

    def calculate_min_distance_to_obstacles(self, position, heading_angle):
        """
        Calculates the minimum distances from the UAV's current position to the nearest obstacle
        for each sensor based on its orientation relative to the UAV's heading.
    
        Parameters:
        - position: The current position of the UAV (numpy array of shape [2]).
        - heading_angle: The current heading angle of the UAV (in radians).
    
        Returns:
        - distances: An array of minimum distances for each sensor.
        """
        distances = np.inf * np.ones(len(self.sensor_angles))  # Initialize distances to infinity
    
        for i, sensor_angle in enumerate(self.sensor_angles):
            # Global orientation of the sensor considering the UAV's heading
            global_sensor_angle = heading_angle + sensor_angle
            
            for obstacle in self.obstacles:
                distance = self.distance_from_point_to_rectangle(position, global_sensor_angle, obstacle)
                distances[i] = min(distances[i], distance)
                # print ("Distance in calculate min dist to obstacle dist[i]:", distances[i])
        
        return distances

        
            
    def calculate_reward_and_done(self, speculative_uav_state):
        done = False
        self.plot = False
        # Extract the position from the speculative state
        current_position = speculative_uav_state[:2]  # Assuming the first two elements are the position
        current_heading_angle = speculative_uav_state[2]
        reward = 0
        current_velocity = speculative_uav_state[3:5]
    
        # Continue with your existing logic, but make sure all references to the UAV's state
        # within this method are replaced with `speculative_uav_state` as needed.
        # _, angle_to_target = self.calculate_distance_and_angle_to_target(current_position)
        # d_dist = -distance_to_target
        # Calculate the reduced distance to the destination
        d_dist = self.calculate_reduced_distance(current_position, self.target_point)
        
        # Calculate the minimum distance to any obstacle
        d_min = self.calculate_min_distance_to_obstacles(current_position, current_heading_angle)
        
        # Determine if the UAV's direction points to free space, consider speculative heading if included in state
        current_heading_angle = speculative_uav_state[2] if len(speculative_uav_state) > 2 else self.uav_state[2]
        free_space = self.check_first_perspective_free_space(current_position)
    
        # Introduce a reward component for choosing a direction with more free space
        best_direction_free_space_reward = 0
        if not free_space:
            # If the direct path is not free, assess alternative directions
            best_direction_angle = self.find_best_free_direction(current_position, current_heading_angle)
            # Convert the best direction angle into a reward, e.g., by relating it to the distance from obstacles
            best_direction_free_space_reward = 0.6 * best_direction_angle
    
        # Calculate each component of the reward
        r_trans = self.sigma * d_dist  # Transition reward
        average_distance = np.min(d_min)
        r_bar = -self.alpha * np.exp(-self.beta * average_distance)
        distance_to_target = np.linalg.norm(current_position - self.target_point)
        # Efficiency penalty (to encourage smooth and efficient movement)

            
        proximity_velocity_penalty = 0
        if distance_to_target <= 1.0:  # Define a range within which to reduce speed
            proximity_velocity_penalty = -0.2 * np.linalg.norm(current_velocity)
        # print(r_bar)
        # r_bar = -self.alpha * np.exp(-self.beta * d_min)  # Obstacle penalty
        r_free = self.r_free if free_space else 0  # Free-space reward
        reward = r_trans + r_bar + r_free + self.r_step + best_direction_free_space_reward + proximity_velocity_penalty # + efficiency_penalty  # Summarize the final non-sparse reward
        # Assume initially no additional done condition
        
        self.previous_velocity = current_velocity
        
        # Calculate energy consumption (simplified)
        u_flow, v_flow = self.get_flow_field_velocity(current_position)
        flow_velocity = np.array([u_flow, v_flow])
        propulsion_velocity = current_velocity - flow_velocity
        energy_penalty = -0.2* np.linalg.norm(propulsion_velocity)
        
        # Add penalties to reward
        reward += energy_penalty
        
        
        if self.has_reached_target(speculative_uav_state[:2]):
            done = True
            self.plot = True
            reward += 50  # Reward for reaching the target
            print("Target reached!")
            return reward, done
        
        if abs(distance_to_target - self.target_radius) < 0.5:
            reward += 0.2
    
        # Checking if the UAV is too close to any obstacle for an additional done condition
        # min_safe_distance_to_obstacle = 0.05  # Example value for minimum safe distance
        # if (average_distance < min_safe_distance_to_obstacle):
        #     # Only consider it too close if all sensor readings are below the safety threshold
        #     done = False
        #     reward += -0.1  # Apply penalty for being too close to an obstacle
        
        out_of_bounds = (
            current_position[0] <= self.domain_x_min or 
            current_position[0] >= self.domain_x_max or 
            current_position[1] <= self.domain_y_min or 
            current_position[1] >= self.domain_y_max
        )
        
        # Apply penalty and end episode if UAV is out of bounds
        if out_of_bounds:
            reward -= 20  # Penalty for being out of bounds
            done = True   # End the episode
            
            # Check for collisions at the speculative next position
        collision = self.check_collision(current_position)    
        if collision:
            # Apply a penalty for collision and potentially end the episode
            reward = -50
            done = True
        # print(f"Current heading angle (rad): {current_heading_angle}")  # Log the current heading angle
        # # Log reward components:
        # print(f"Transition reward (r_trans): {r_trans}")
        # print(f"Obstacle penalty (r_bar): {r_bar}")
        # print(f"Free-space reward (r_free): {r_free}")
        # print(f"Best direction free space reward: {best_direction_free_space_reward}")
        # If none of the above conditions are met and the UAV is still operational
        if not done and not collision and not out_of_bounds:
             self.uav_state = speculative_uav_state

        return reward, done

    
    def has_reached_target(self, current_position):
        # Calculate the distance to the target point
        distance_to_target = np.linalg.norm(current_position - self.target_point)
        
        # Define a margin of error, considering the UAV's positional accuracy
        margin_of_error = 0.05  # Adjust the margin as needed for your system
        # Check if the UAV is within the margin of error of the target radius
        return abs(distance_to_target - self.target_radius) <= margin_of_error