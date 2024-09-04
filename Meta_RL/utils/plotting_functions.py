#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 20:35:34 2024

@author: federicatonti
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.interpolate import splprep, splev
import h5py
import numpy as np



def plot_environment_with_uav(combined_file_path, timestep, obstacles, uav_positions, obs_min, obs_max, start_point, start_radius, target_area_center, target_radius):
    """
    Plots the velocity field from the simulation at a given timestep along with the UAV's trajectory.

    Parameters:
    - combined_file_path: Path to the HDF5 file containing the simulation data.
    - timestep: Current timestep of the simulation to plot.
    - obstacles: List of tuples representing obstacle positions and sizes.
    - uav_positions: List of tuples representing the UAV's positions over time.
    - plots_folder: Folder to save the generated plots.
    """
    # Ensure the plots folder exists
    # if not os.path.exists(plots_folder):
    #     os.makedirs(plots_folder)

    # Open the combined HDF5 file
    with h5py.File(combined_file_path, 'r') as hdf:
        # Read the X, Y data and U velocity for the current timestep
        X = hdf['X'][:]
        Y = hdf['Y'][:]
        U_timestep = hdf['U'][timestep, :, :]
        
        # Find global minimum and maximum for U velocity for normalization
        vmin = np.min(U_timestep)
        vmax = np.max(U_timestep)

        fig, ax = plt.subplots(figsize=(10, 6))
        mesh = ax.pcolormesh(X, Y, U_timestep, cmap='viridis', vmin=vmin, vmax=vmax)
        fig.colorbar(mesh, ax=ax, label='U Velocity')

        # Overlay obstacles
        for obs in obstacles:
            rect = patches.Rectangle((obs[0], obs[1]), obs[2], obs[3], linewidth=1, edgecolor='r', facecolor='red', alpha=0.5)
            ax.add_patch(rect)
            
        start_circle = plt.Circle(start_point, start_radius, color='green', fill=False)
        ax.add_patch(start_circle)
    
        # Overlay target region
        target_circle = plt.Circle(target_area_center, target_radius, color='blue', fill=False)
        ax.add_patch(target_circle)


        for pos in uav_positions:
            # Apply the denormalization
            actual_pos = (np.array(pos) * (obs_max - obs_min)) + obs_min
            # print(actual_pos)
            ax.plot(actual_pos[0], actual_pos[1], color='red', marker='+', linestyle='dashed',
     linewidth=1, markersize=8)
            

        ax.set_title(f'U Velocity and UAV Position at Time Step {timestep+1}')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        plt.xlim(np.min(X), np.max(X))
        plt.ylim(np.min(Y), np.max(Y))

        # Save the plot
        # plot_path = os.path.join(plots_folder, f'U_velocity_UAV_position_{timestep+1}.png')
        # plt.savefig(plot_path)
        plt.show()
        plt.close(fig)  # Close the figure to free memory
        
        
def plot_trajectory_over_environment(h5_file_path, timestep, trajectory, obstacles, obs_min, obs_max, start_point, start_radius, target_area_center, target_radius, episode_number = None):
    with h5py.File(h5_file_path, 'r') as hdf:
        # Load data for the current timestep
        X = hdf['X'][:]
        Y = hdf['Y'][:]
        U = hdf['U'][timestep, :, :]
        V = hdf['V'][timestep, :, :]
        abs_velocity = np.sqrt(U**2 + V**2)

        # Setup the figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))
        mesh = ax.pcolormesh(X, Y, abs_velocity, cmap='viridis', vmin=0, vmax=1.8)
        fig.colorbar(mesh, ax=ax, label='Absolute Velocity')

        # Overlay obstacles
        for obs in obstacles:
            rect = patches.Rectangle((obs[0], obs[1]), obs[2], obs[3], linewidth=1, edgecolor='r', facecolor='red', alpha=0.5)
            ax.add_patch(rect)

        # Plot the start point and target area
        start_circle = patches.Circle(start_point, start_radius, color='y', fill=False, label='Start Point')
        ax.add_patch(start_circle)

        target_circle = patches.Circle(target_area_center, target_radius, color='m', fill=False, label='Target Area')
        ax.add_patch(target_circle)

        # Extract trajectory points
        traj_x, traj_y = zip(*trajectory)
        # Check for duplicates or non-increasing values in traj_x

        # t = np.linspace(0, 1, len(traj_x))
        # tck, u = splprep([traj_x, traj_y], u=t, s=0.5)
        # unew = np.linspace(0, 1, 200)
        
        # xnew, ynew = splev(unew, tck)


        # Plot the UAV trajectory
        ax.plot(traj_x, traj_y, 'r-', markersize=2, zorder=5, label='UAV Trajectory')  # Original trajectory points
        # ax.plot(xnew, ynew, 'r-', linewidth=2, zorder=6, label='UAV Trajectory')  # Smoothed trajectory

        # Set plot labels and title
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title(f'Absolute Velocity and UAV Trajectory at Timestep {timestep + 1}')
        ax.legend()
        plt.show()
        plt.close()
        


def plot_PPO(X, Y, U, V, trajectory, obstacles, start_point, start_radius, target_area_center, target_radius, episode_number, save_dir='../plots_PPO_lin_ang'):
    """
    Plot the absolute velocity field and UAV trajectory over an environment.
    
    Parameters:
    - X, Y: Arrays representing the grid coordinates for the plot.
    - U, V: Arrays representing the velocity components at each grid point.
    - trajectory: List of tuples, each representing the (x, y) coordinates of the UAV at each step.
    - obstacles: List of tuples, each representing an obstacle in the format (x, y, width, height).
    - start_point: Tuple (x, y) representing the starting point of the UAV.
    - start_radius: Float representing the radius of the circle to draw for the start point.
    - target_area_center: Tuple (x, y) representing the center of the target area.
    - target_radius: Float representing the radius of the circle to draw for the target area.
    """
    abs_velocity = np.sqrt(U**2 + V**2)

    # Setup the figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    mesh = ax.pcolormesh(X, Y, abs_velocity, cmap='viridis', shading='auto', vmin=0, vmax=1.8)
    fig.colorbar(mesh, ax=ax, label='Absolute Velocity')

    # Overlay obstacles
    for obs in obstacles:
        rect = patches.Rectangle((obs[0], obs[1]), obs[2], obs[3], linewidth=1, edgecolor='r', facecolor='red', alpha=0.5)
        ax.add_patch(rect)

    # Plot the start point and target area
    start_circle = patches.Circle(start_point, start_radius, color='y', fill=False, label='Start Point')
    ax.add_patch(start_circle)
    target_circle = patches.Circle(target_area_center, target_radius, color='m', fill=False, label='Target Area')
    ax.add_patch(target_circle)

    # Plot the UAV trajectory
    traj_x, traj_y = zip(*trajectory)
    ax.plot(traj_x, traj_y, 'r-', markersize=2, zorder=5, label='UAV Trajectory')

    # Set plot labels and title
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    plt.xlim(np.min(X), np.max(X))
    plt.ylim(np.min(Y), np.max(Y))
    ax.set_title('Absolute Velocity and UAV Trajectory Visualization')
    ax.legend()

# Save the plot if episode number is provided
    filename = f"{save_dir}/trajectory_episode_{episode_number}.png"
    plt.savefig(filename, bbox_inches='tight', dpi=300)

    # plt.show()
    plt.close(fig)  # Close the figure to free memory


def plot_TD3(X, Y, U, V, trajectory, obstacles, start_point, start_radius, target_area_center, target_radius, episode_number, save_dir='plots_TD3'):
    """
    Plot the absolute velocity field and UAV trajectory over an environment.
    
    Parameters:
    - X, Y: Arrays representing the grid coordinates for the plot.
    - U, V: Arrays representing the velocity components at each grid point.
    - trajectory: List of tuples, each representing the (x, y) coordinates of the UAV at each step.
    - obstacles: List of tuples, each representing an obstacle in the format (x, y, width, height).
    - start_point: Tuple (x, y) representing the starting point of the UAV.
    - start_radius: Float representing the radius of the circle to draw for the start point.
    - target_area_center: Tuple (x, y) representing the center of the target area.
    - target_radius: Float representing the radius of the circle to draw for the target area.
    """
    abs_velocity = np.sqrt(U**2 + V**2)

    # Setup the figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    mesh = ax.pcolormesh(X, Y, abs_velocity, cmap='viridis', shading='auto', vmin=0, vmax=1.8)
    fig.colorbar(mesh, ax=ax, label='Absolute Velocity')

    # Overlay obstacles
    for obs in obstacles:
        rect = patches.Rectangle((obs[0], obs[1]), obs[2], obs[3], linewidth=1, edgecolor='r', facecolor='red', alpha=0.5)
        ax.add_patch(rect)

    # Plot the start point and target area
    start_circle = patches.Circle(start_point, start_radius, color='y', fill=False, label='Start Point')
    ax.add_patch(start_circle)
    target_circle = patches.Circle(target_area_center, target_radius, color='m', fill=False, label='Target Area')
    ax.add_patch(target_circle)

    # Plot the UAV trajectory
    traj_x, traj_y = zip(*trajectory)
    ax.plot(traj_x, traj_y, 'r-', markersize=2, zorder=5, label='UAV Trajectory')

    # Set plot labels and title
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    plt.xlim(np.min(X), np.max(X))
    plt.ylim(np.min(Y), np.max(Y))
    ax.set_title('Absolute Velocity and UAV Trajectory Visualization')
    ax.legend()

# Save the plot if episode number is provided
    filename = f"{save_dir}/trajectory_episode_{episode_number}.png"
    plt.savefig(filename, bbox_inches='tight', dpi=300)

    #plt.show()
    plt.close(fig)  # Close the figure to free memory
    
def plot_DDPG(X, Y, U, V, trajectory, obstacles, start_point, start_radius, target_area_center, target_radius, episode_number, save_dir='plots_DDPG'):
    """
    Plot the absolute velocity field and UAV trajectory over an environment.
    
    Parameters:
    - X, Y: Arrays representing the grid coordinates for the plot.
    - U, V: Arrays representing the velocity components at each grid point.
    - trajectory: List of tuples, each representing the (x, y) coordinates of the UAV at each step.
    - obstacles: List of tuples, each representing an obstacle in the format (x, y, width, height).
    - start_point: Tuple (x, y) representing the starting point of the UAV.
    - start_radius: Float representing the radius of the circle to draw for the start point.
    - target_area_center: Tuple (x, y) representing the center of the target area.
    - target_radius: Float representing the radius of the circle to draw for the target area.
    """
    abs_velocity = np.sqrt(U**2 + V**2)

    # Setup the figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    mesh = ax.pcolormesh(X, Y, abs_velocity, cmap='viridis', shading='auto', vmin=0, vmax=1.8)
    fig.colorbar(mesh, ax=ax, label='Absolute Velocity')

    # Overlay obstacles
    for obs in obstacles:
        rect = patches.Rectangle((obs[0], obs[1]), obs[2], obs[3], linewidth=1, edgecolor='r', facecolor='red', alpha=0.5)
        ax.add_patch(rect)

    # Plot the start point and target area
    start_circle = patches.Circle(start_point, start_radius, color='y', fill=False, label='Start Point')
    ax.add_patch(start_circle)
    target_circle = patches.Circle(target_area_center, target_radius, color='m', fill=False, label='Target Area')
    ax.add_patch(target_circle)

    # Plot the UAV trajectory
    traj_x, traj_y = zip(*trajectory)
    ax.plot(traj_x, traj_y, 'r-', markersize=2, zorder=5, label='UAV Trajectory')

    # Set plot labels and title
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    plt.xlim(np.min(X), np.max(X))
    plt.ylim(np.min(Y), np.max(Y))
    ax.set_title('Absolute Velocity and UAV Trajectory Visualization')
    ax.legend()

# Save the plot if episode number is provided
    filename = f"{save_dir}/trajectory_episode_{episode_number}.png"
    plt.savefig(filename, bbox_inches='tight', dpi=300)

    plt.show()
    plt.close(fig)  # Close the figure to free memory




            
