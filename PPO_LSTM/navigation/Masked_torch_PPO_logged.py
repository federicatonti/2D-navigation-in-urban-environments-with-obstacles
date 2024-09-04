#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 09:34:08 2024

@author: federicatonti
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from gymnasium.wrappers import NormalizeObservation, NormalizeReward
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from PIL import Image



def get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "tanh":
        return torch.tanh
    elif activation == "sigmoid":
        return torch.sigmoid
    else:
        raise ValueError("Unknown activation function: {}".format(activation))

class PPO_LSTM(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim1=64, hidden_dim2=32, lstm_hidden_dim_recurrent=16, encoder_activation='tanh', vf_hiddens=[64, 64], vf_activation='tanh'):
        super(PPO_LSTM, self).__init__()#h1=128,h2=64,lstm_hidd=31 #h1=60,h2=30,lstm_hidd=15

        # MLP Encoder Layers
        self.encoder_activation_fn = get_activation_fn(encoder_activation)
        self.fc1 = nn.Linear(obs_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)

        # LSTM Layer
        self.lstm = nn.LSTM(hidden_dim2 + act_dim + 1, lstm_hidden_dim_recurrent, batch_first=True)
        
        # Output Layers
        self.fc_mean = nn.Linear(lstm_hidden_dim_recurrent, act_dim)
        self.fc_std = nn.Linear(lstm_hidden_dim_recurrent, act_dim)
        self.fc_value = nn.Linear(lstm_hidden_dim_recurrent, 1)  # Value head for the critic

        # Parameters
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.lstm_hidden_dim_recurrent = lstm_hidden_dim_recurrent

    def forward(self, obs, last_act, last_rew, hidden_state, mask):
        batch_size = obs.size(0)
        seq_len = obs.size(1) if obs.dim() == 3 else 1

        # MLP Encoder Forward Pass
        x = obs
        x = self.encoder_activation_fn(self.fc1(x))
        x = self.encoder_activation_fn(self.fc2(x))

        # Ensure mask is the same shape as x
        if mask.dim() == 2:
            mask = mask.unsqueeze(-1)

        # Apply the mask
        x = x * mask

        # Handle dimensions of last_act and last_rew
        if last_act.dim() == 3 and seq_len > 1:
            last_act = last_act.expand(-1, seq_len, -1) * mask
        elif last_act.dim() == 2:
            last_act = last_act.unsqueeze(1).expand(-1, seq_len, -1) * mask
        else:
            last_act = last_act * mask

        if last_rew.dim() == 3 and seq_len > 1:
            last_rew = last_rew.expand(-1, seq_len, -1) * mask
        elif last_rew.dim() == 2:
            last_rew = last_rew.unsqueeze(1).expand(-1, seq_len, -1) * mask
        else:
            last_rew = last_rew * mask

        lstm_input = torch.cat([x, last_act, last_rew], dim=-1)
        lstm_output, hidden_state = self.lstm(lstm_input, hidden_state)

        # Reshape LSTM output to match fully connected layer input
        lstm_output = lstm_output.contiguous().view(-1, lstm_output.size(2))

        # Get action mean, std, and value from the LSTM output
        action_mean = self.fc_mean(lstm_output)
        action_std = F.softplus(self.fc_std(lstm_output))  # Ensure positive std
        value = self.fc_value(lstm_output)

        return action_mean, action_std, value, hidden_state

    def init_hidden_state(self, batch_size):
        return (torch.zeros(1, batch_size, self.lstm_hidden_dim_recurrent),
                torch.zeros(1, batch_size, self.lstm_hidden_dim_recurrent))
    


class PPOTrainer:
    def __init__(self, model, optimizer, clip_ratio=0.2, epochs=3000, sgd_steps=30, gamma=0.99, lam=0.95, mini_batch_size=128, k_epochs=4, eval_interval = 5):  #gamma=0.99, lam=0.95,
        self.model = model
        self.optimizer = optimizer
        self.clip_ratio = clip_ratio
        self.epochs = epochs
        self.sgd_steps = sgd_steps
        self.gamma = gamma
        self.lam = lam
        self.mini_batch_size = mini_batch_size
        self.k_epochs = k_epochs
        self.eval_interval = eval_interval
        self.writer = SummaryWriter(log_dir=f"runs/{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        self.best_reward = -float('inf')
        self.best_model_state = None
        self.episode_count = 0
        self.update_count = 0  # Initialize update count
        
    def save_checkpoint(self, filename):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_reward': self.best_reward,
            'best_model_state': self.best_model_state,
            'update_count': self.update_count,
            'episode_count': self.episode_count,
        }
        torch.save(checkpoint, filename)
        print(f"Checkpoint saved to {filename}")
    
    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_reward = checkpoint['best_reward']
        self.best_model_state = checkpoint['best_model_state']
        self.update_count = checkpoint['update_count']
        self.episode_count = checkpoint['episode_count']
        print(f"Checkpoint loaded from {filename}")

    def compute_gae(self, rewards, values, dones, gamma=0.99, lam=0.95):
        assert isinstance(rewards, torch.Tensor), "rewards should be a PyTorch tensor"
        assert isinstance(values, torch.Tensor), "values should be a PyTorch tensor"
        assert isinstance(dones, torch.Tensor), "dones should be a PyTorch tensor"

        rewards = rewards.numpy()
        values = values.numpy()
        dones = dones.numpy()

        batch_size, seq_len, _ = rewards.shape

        all_advantages = np.zeros((batch_size, seq_len))
        all_returns = np.zeros((batch_size, seq_len))

        for i in range(batch_size):
            gae = 0
            rewards_batch = rewards[i, :, 0]
            values_batch = values[i, :, 0]
            dones_batch = dones[i, :, 0]

            for step in reversed(range(seq_len)):
                if step == seq_len - 1:
                    next_value = values_batch[step]
                    next_done = dones_batch[step]
                else:
                    next_value = values_batch[step + 1]
                    next_done = dones_batch[step + 1]

                delta = rewards_batch[step] + gamma * next_value * (1 - next_done) - values_batch[step]
                gae = delta + gamma * lam * (1 - next_done) * gae
                all_advantages[i, step] = gae
                all_returns[i, step] = gae + values_batch[step]

        return torch.tensor(all_advantages, dtype=torch.float32), torch.tensor(all_returns, dtype=torch.float32)
    
    def pad_and_stack(self, sequences, padding_value=-1):
        # Check if all elements in sequences are lists or tensors
        if not all(isinstance(seq, (list, torch.Tensor)) for seq in sequences):
            raise TypeError("All elements of `sequences` must be lists or tensors")
    
        # Calculate lengths of each sequence
        lengths = [len(seq) if isinstance(seq, (list, torch.Tensor)) else 1 for seq in sequences]
        max_len = max(lengths, default=0)
    
        # Initialize padded sequences list
        padded_seqs = []
        for seq in sequences:
            if isinstance(seq, int):
                padded_seqs.append([seq] * max_len)
            elif isinstance(seq, list) and len(seq) == 0:
                inner_len = 1
                padded_seqs.append([padding_value] * max_len)
            elif isinstance(seq[0], (list, torch.Tensor)):
                inner_len = len(seq[0])
                padding_needed = max_len - len(seq)
                padded_seqs.append(seq + [[padding_value] * inner_len] * padding_needed)
            else:
                inner_len = 1
                padding_needed = max_len - len(seq)
                padded_seqs.append(seq + [padding_value] * padding_needed)
    
        # Convert to tensor
        padded_seqs_tensor = torch.tensor(padded_seqs, dtype=torch.float32)
    
        # Debugging: Print the padded sequences
        # print("Padded Sequences:")
        # print(padded_seqs_tensor)
        # print(f"Expected padding value: {padding_value}")
        # print(f"Lengths: {lengths}")
    
        # Check if the padding value is correctly used
        total_padding_elements = sum((max_len - length) * (len(seq[0]) if isinstance(seq[0], (list, torch.Tensor)) else 1) for length, seq in zip(lengths, sequences))
        # print("Total padding elements:", total_padding_elements)
    
        # Explicitly count padding values
        padding_count = (padded_seqs_tensor == padding_value).sum().item()
    
        if padding_count != total_padding_elements:
            print(f"Error: Padding value not correctly applied. Expected {total_padding_elements} padding values but found {padding_count}.")
    
        return padded_seqs_tensor, lengths
 
    
    def compute_loss(self, states, last_rewards, last_actions, actions, old_log_probs, advantages, returns, mask):
        batch_size = states.size(0)
        seq_len = states.size(1)
        hidden_state = self.model.init_hidden_state(batch_size)
    
        # Check for NaN in inputs
        def check_for_nans(tensor, name):
            if torch.isnan(tensor).any():
                print(f'NaN detected in {name}')
        
        check_for_nans(states, 'states')
        check_for_nans(last_rewards, 'last_rewards')
        check_for_nans(last_actions, 'last_actions')
        check_for_nans(actions, 'actions')
        check_for_nans(old_log_probs, 'old_log_probs')
        check_for_nans(advantages, 'advantages')
        check_for_nans(returns, 'returns')
        check_for_nans(mask, 'mask')
    
        # Forward pass through the model
        action_means, action_stds, values, _ = self.model(states, last_actions, last_rewards, hidden_state, mask)
    
        check_for_nans(action_means, 'action_means')
        check_for_nans(action_stds, 'action_stds')
        check_for_nans(values, 'values')
    
        # Reshape to ensure proper broadcasting
        action_means = action_means.view(batch_size, seq_len, -1)
        action_stds = action_stds.view(batch_size, seq_len, -1)
        values = values.view(batch_size, seq_len, 1)  # Ensure values has the same shape as returns
    
        # Reshape log_probs to match actions' shape
        action_dists = Normal(action_means, action_stds)
        log_probs = action_dists.log_prob(actions).sum(-1)
        check_for_nans(log_probs, 'log_probs')
    
        # Apply mask to log_probs and values
        mask_expanded = mask.unsqueeze(-1)  # Make mask the same shape as values
        log_probs = log_probs * mask
        values = values * mask_expanded
    
        # Debug log_probs shape
        # print(f'log_probs shape: {log_probs.shape}')
        # print(f'values shape: {values.shape}')
        # print(f'returns shape: {returns.shape}')
    
        if log_probs.shape != old_log_probs.shape:
            old_log_probs = old_log_probs.squeeze(-1)
    
        ratios = torch.exp(log_probs - old_log_probs)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
    
        check_for_nans(policy_loss, 'policy_loss')
    
        # if values.shape != returns.shape:
        #     values = values.squeeze(-1)
    
        # Apply mask to returns and advantages before computing the loss
        returns = returns.unsqueeze(-1) * mask_expanded
        advantages = advantages.unsqueeze(-1) * mask_expanded
    
        value_loss = ((returns - values) ** 2).mean()
        check_for_nans(value_loss, 'value_loss')
    
        entropy_bonus = action_dists.entropy().mean()
        check_for_nans(entropy_bonus, 'entropy_bonus')
    
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_bonus
        check_for_nans(loss, 'loss')
    
        return loss

    def update(self, trajectories, mini_batch_size):
        # Unpack trajectories
        states = trajectories['states']
        actions = trajectories['actions']
        rewards = trajectories['rewards']
        next_states = trajectories['next_states']
        values = trajectories['values']
        dones = trajectories['dones']
        log_probs = trajectories['log_probs']
        last_actions = trajectories['last_actions']
        last_rewards = trajectories['last_rewards']
        masks = trajectories['masks']
    
        # Pad and stack trajectories
        states_padded, _ = self.pad_and_stack(states, padding_value=-10)
        actions_padded, _ = self.pad_and_stack(actions, padding_value=-10)
        rewards_padded, _ = self.pad_and_stack(rewards, padding_value=-10)
        dones_padded, _ = self.pad_and_stack(dones, padding_value=1.5)
        values_padded, _ = self.pad_and_stack(values, padding_value=-10)
        log_probs_padded, _ = self.pad_and_stack(log_probs, padding_value=-10)
        last_actions_padded, _ = self.pad_and_stack(last_actions, padding_value=-10)
        last_rewards_padded, _ = self.pad_and_stack(last_rewards, padding_value=-10)
        masks_padded, _ = self.pad_and_stack(masks, padding_value=0.0)
    
        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards_padded, values_padded, dones_padded)
    
        # Print to ensure padding and GAE computation are correct
        print(f"States Padded: {states_padded.shape}")
        print(f"Actions Padded: {actions_padded.shape}")
        print(f"Rewards Padded: {rewards_padded.shape}")
        print(f"Dones Padded: {dones_padded.shape}")
        print(f"Values Padded: {values_padded.shape}")
        print(f"Log Probs Padded: {log_probs_padded.shape}")
        print(f"Last Actions Padded: {last_actions_padded.shape}")
        print(f"Last Rewards Padded: {last_rewards_padded.shape}")
        print(f"Masks Padded: {masks_padded.shape}")
        print(f"Advantages: {advantages.shape}")
        print(f"Returns: {returns.shape}")
    
        total_loss = 0
        data_size = states_padded.size(0)
        indices = np.arange(data_size)
    
        # Loop over epochs and SGD steps
        for k in range(self.k_epochs):
            print(f"Epoch {k+1}/{self.k_epochs}")
            for epoch in range(self.sgd_steps):
                np.random.shuffle(indices)
                for start in range(0, data_size, mini_batch_size):
                    end = min(start + mini_batch_size, data_size)
                    mini_batch_indices = indices[start:end]
    
                    mini_batch_states = states_padded[mini_batch_indices]
                    mini_batch_last_rewards = last_rewards_padded[mini_batch_indices]
                    mini_batch_last_actions = last_actions_padded[mini_batch_indices]
                    mini_batch_actions = actions_padded[mini_batch_indices]
                    mini_batch_log_probs = log_probs_padded[mini_batch_indices]
                    mini_batch_advantages = advantages[mini_batch_indices]
                    mini_batch_returns = returns[mini_batch_indices]
                    mini_batch_masks = masks_padded[mini_batch_indices]
    
                    # Compute loss
                    loss = self.compute_loss(
                        mini_batch_states,
                        mini_batch_last_rewards,
                        mini_batch_last_actions,
                        mini_batch_actions,
                        mini_batch_log_probs,
                        mini_batch_advantages,
                        mini_batch_returns,
                        mini_batch_masks
                    )
    
                    # Backpropagation and optimization
                    self.optimizer.zero_grad()
                    loss.backward()
    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
    
                    total_loss += loss.item()
    
                    # Print loss for each mini-batch for debugging
                    print(f"Mini-batch {start}-{end}, Loss: {loss.item()}")
    
        # Increment the update count
        self.update_count += 1
    
        # Log the average loss
        avg_loss = total_loss / (self.k_epochs * self.sgd_steps * (data_size / mini_batch_size))
        self.writer.add_scalar("Loss/AvgLoss", avg_loss, self.update_count)
        print("Average Loss:", avg_loss)
    
        # Log the mean returns for the batch
        mean_return = returns.mean().item()
        self.writer.add_scalar("Returns/MeanReturns", mean_return, self.update_count)
        print("Mean Return:", mean_return)
    
        # Update the best model state if the current mean return is the best seen so far
        if mean_return > self.best_reward:
            self.best_reward = mean_return
            self.best_model_state = self.model.state_dict()
    
        # Load the best model state
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        

                    
    def train(self, env, checkpoint_interval=1, checkpoint_path='checkpoints'):
        episode_count = 0  # Initialize episode count
        # Load the best model state if it exists
        
        for epoch in range(self.epochs):
            # Collect trajectories, passing the current episode count
            trajectories = self.collect_trajectories(env, self.model, batch_size=256, episode_start_count=episode_count)
            self.update(trajectories, self.mini_batch_size)
            
            # Update the episode count
            batch_size = 256
            episode_count += batch_size
            
            # Evaluate the policy
            if (epoch + 1) % self.eval_interval == 0:
                self.evaluate_policy(env)
            
            # Save a checkpoint
            if (epoch + 1) % checkpoint_interval == 0:
                checkpoint_filename = f'checkpoint_epoch_{epoch + 1}.pth'
                self.save_checkpoint(checkpoint_filename)
                
        self.writer.close()



    def evaluate_policy(self, env, episodes=50, max_timesteps=81):
        total_rewards = []
        for episode in range(episodes):
            state, _ = env.reset()
            done = False
            steps = 0
            cumulative_reward = 0
            hidden_state = self.model.init_hidden_state(1)
            last_action = np.zeros((1, 1, env.action_space.shape[0]))
            last_reward = np.array([[[0.0]]])
 
            while not done and steps < max_timesteps:
                obs_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                last_action_tensor = torch.tensor(last_action, dtype=torch.float32)
                last_reward_tensor = torch.tensor(last_reward, dtype=torch.float32)
                mask_tensor = torch.tensor([1], dtype=torch.float32).unsqueeze(0)

                with torch.no_grad():
                    action_mean, action_std, value, hidden_state = self.model(obs_tensor, last_action_tensor, last_reward_tensor, hidden_state, mask_tensor)
                action_dist = Normal(action_mean, action_std)
                action = action_dist.sample()
                action_np = action.numpy().flatten()
                next_state, reward, done, truncated, _ = env.step(action_np)
                cumulative_reward += reward

                state = next_state
                last_action = action_np.reshape(1, 1, -1)
                last_reward = np.array([[[reward]]])
                steps += 1

            total_rewards.append(cumulative_reward)
            print(f"Evaluation Episode {episode + 1}: Total Reward = {cumulative_reward:.2f}")
        
        avg_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        print(f"Average Reward over {episodes} Evaluation Episodes: {avg_reward:.2f} ± {std_reward:.2f}")
        self.log_results(avg_reward, std_reward)
        self.save_best_model(avg_reward)
        
    def log_results(self, avg_reward, std_reward):
        # Log results to TensorBoard or any other logging tool
        self.writer.add_scalar("Evaluation/AvgReward", avg_reward, self.update_count)
        self.writer.add_scalar("Evaluation/StdReward", std_reward, self.update_count)
        print(f"Logged AvgReward: {avg_reward:.2f} and StdReward: {std_reward:.2f} to TensorBoard")
        
    def save_best_model(self, avg_reward):
        # Save the best model based on average reward
        if avg_reward > self.best_reward:
            self.best_reward = avg_reward
            self.best_model_state = self.model.state_dict()
            torch.save(self.best_model_state, 'best_model.pth')
            print(f"New best model saved with average reward: {avg_reward:.2f}")
            
    def plot_actions(self, actions, episode_num):
        """
        Plot a scatter plot, histogram, and heat map of the actions taken by the agent,
        and log them to TensorBoard.
        """
        # print(f"Plotting actions for episode {episode_num}...")

        actions = np.array(actions)  # Convert to numpy array for easier processing

        # Ensure actions have the correct shape
        if len(actions.shape) == 3:
            actions = actions.reshape(-1, actions.shape[-1])
        # print(f"Actions shape after reshaping: {actions.shape}")

        # Scatter plot
        fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6))
        ax_scatter.scatter(actions[:, 0], actions[:, 1], alpha=0.5, s=10)
        ax_scatter.set_title(f'Scatter Plot of Actions - Episode {episode_num}')
        ax_scatter.set_xlabel('Action Dimension 1')
        ax_scatter.set_ylabel('Action Dimension 2')
        ax_scatter.grid(True)
        # print("Scatter plot created.")

        # Histogram of actions
        fig_hist, ax_hist = plt.subplots(figsize=(10, 6))
        ax_hist.hist(actions, bins=30, alpha=0.7, label=['Action Dim 1', 'Action Dim 2'])
        ax_hist.set_title(f'Histogram of Actions - Episode {episode_num}')
        ax_hist.set_xlabel('Action Value')
        ax_hist.set_ylabel('Frequency')
        ax_hist.legend()
        ax_hist.grid(True)
        # print("Histogram created.")

        # Heat map
        fig_heatmap, ax_heatmap = plt.subplots(figsize=(8, 6))
        sns.heatmap(np.histogram2d(actions[:, 0], actions[:, 1], bins=50)[0], cmap='viridis', ax=ax_heatmap)
        ax_heatmap.set_title(f'Heat Map of Actions - Episode {episode_num}')
        ax_heatmap.set_xlabel('Action Dimension 1')
        ax_heatmap.set_ylabel('Action Dimension 2')
        # print("Heatmap created.")

        # Log the plots to TensorBoard
        self._log_plot_to_tensorboard(fig_scatter, f'ScatterPlot/Episode{episode_num}', episode_num)
        # print("Scatter plot logged to TensorBoard.")
        self._log_plot_to_tensorboard(fig_hist, f'Histogram/Episode{episode_num}', episode_num)
        # print("Histogram logged to TensorBoard.")
        self._log_plot_to_tensorboard(fig_heatmap, f'Heatmap/Episode{episode_num}', episode_num)
        # print("Heatmap logged to TensorBoard.")

        # Close figures to avoid memory leaks
        plt.close(fig_scatter)
        plt.close(fig_hist)
        plt.close(fig_heatmap)
        # print("Figures closed.")

    def _log_plot_to_tensorboard(self, fig, tag, step):
        """
        Convert a Matplotlib figure to an image and log it to TensorBoard.
        """
        # print(f"Logging {tag} to TensorBoard at step {step}...")
        buf = BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)  # Open the image with PIL
        image = np.array(image)  # Convert the PIL image to a numpy array
        self.writer.add_image(tag, image, global_step=step, dataformats='HWC')
        buf.close()
        # print(f"{tag} logged successfully.")


    def collect_trajectories(self, env, model, batch_size, plot=True, episode_start_count=0):
        trajectories = {
            'states': [], 'actions': [], 'rewards': [], 'next_states': [],
            'values': [], 'dones': [], 'log_probs': [],
            'last_actions': [], 'last_rewards': [], 'masks': []
        }
    
        episode_num = episode_start_count  # Start from the passed episode count
    
        for _ in range(batch_size):
            state, _ = env.reset()
            last_action = np.zeros((1, 1, env.action_space.shape[0]))  
            last_reward = np.array([[[0.0]]])  
            mask = []
    
            done = False
            steps = 0
            cumulative_reward = 0
            episode_states = []
            episode_actions = []
            episode_rewards = []
            episode_next_states = []
            episode_values = []
            episode_dones = []
            episode_log_probs = []
            episode_last_actions = []
            episode_last_rewards = []
            episode_masks = []
            
            hidden_state = model.init_hidden_state(1) 
    
            while not done:
                # Convert to tensor
                obs_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  
                last_action_tensor = torch.tensor(last_action, dtype=torch.float32)  
                last_reward_tensor = torch.tensor(last_reward, dtype=torch.float32)
                mask_tensor = torch.tensor([1], dtype=torch.float32).unsqueeze(0)
    
                with torch.no_grad():
                    action_mean, action_std, value, hidden_state = model(obs_tensor, last_action_tensor, last_reward_tensor, hidden_state, mask_tensor)
                action_dist = Normal(action_mean, action_std)
                action = action_dist.sample()  
                log_prob = action_dist.log_prob(action).sum(dim=-1)
    
                action_np = action.numpy().flatten()
                next_state, reward, done, truncated, _ = env.step(action_np)
                cumulative_reward += reward
    
                episode_states.append(state.tolist())
                episode_actions.append(action_np.tolist())
                episode_rewards.append([reward])  # Make reward a list
                episode_next_states.append(next_state.tolist())
                episode_values.append([value.item()])  # Make value a list
                episode_dones.append([done])  # Make done a list
                episode_log_probs.append([log_prob.item()])  # Make log_prob a list
                episode_last_actions.append(last_action.flatten().tolist())
                episode_last_rewards.append(last_reward.flatten().tolist())
                mask.append(1)  # Add 1 to mask for valid data points
    
                state = next_state
                last_action = action_np.reshape(1, 1, -1)
                last_reward = np.array([[[reward]]])
                steps += 1
                
                # if plot:
                #     plot_PPO(
                #         X=np.array(env.flow_field_data['X']),
                #         Y=np.array(env.flow_field_data['Y']),
                #         U=np.array(env.flow_field_data['U'][env.current_time_step, :, :]),
                #         V=np.array(env.flow_field_data['V'][env.current_time_step, :, :]),
                #         trajectory=env.trajectory,
                #         obstacles=[(-0.25, 0, 0.5, 1), (1.25, 0, 0.5, 0.5)],
                #         start_point=env.start_point,
                #         start_radius=env.start_radius,
                #         target_area_center=env.target_area_center,
                #         target_radius=env.target_radius,
                #         episode_number=episode_num
                #     )
        
    
            trajectories['states'].append(episode_states)
            trajectories['actions'].append(episode_actions)
            trajectories['rewards'].append(episode_rewards)
            trajectories['next_states'].append(episode_next_states)
            trajectories['values'].append(episode_values)
            trajectories['dones'].append(episode_dones)
            trajectories['log_probs'].append(episode_log_probs)
            trajectories['last_actions'].append(episode_last_actions)
            trajectories['last_rewards'].append(episode_last_rewards)
            trajectories['masks'].append(mask)
    
            # Log cumulative reward for this episode to TensorBoard
            self.writer.add_scalar("Reward/Episode", cumulative_reward, episode_num)
    
            print(f"Episode {episode_num} finished after {steps} steps with total reward {cumulative_reward:.2f}.")
            if steps == 0 or cumulative_reward == 0:
                print("Warning: Collected an empty episode.")
                return None
    
            if plot:
                plot_PPO(
                    X=np.array(env.flow_field_data['X']),
                    Y=np.array(env.flow_field_data['Y']),
                    U=np.array(env.flow_field_data['U'][env.current_time_step, :, :]),
                    V=np.array(env.flow_field_data['V'][env.current_time_step, :, :]),
                    trajectory=env.trajectory,
                    obstacles=[(-0.25, 0, 0.5, 1), (1.25, 0, 0.5, 0.5)],
                    start_point=env.start_point,
                    start_radius=env.start_radius,
                    target_area_center=env.target_area_center,
                    target_radius=env.target_radius,
                    episode_number=episode_num
                )
                
                self.plot_actions(episode_actions, episode_num)   
    
            episode_num += 1  # Increment the episode number for the next iteration
    
        return trajectories




import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from environment.zermelos_like_angle_PPO_2 import FlowFieldNavEnv
from environment.head_vel  import FlowFieldNavEnv
# from environment.head_vel import FlowFieldNavEnv
from utils.plotting_functions import plot_PPO

# Usage
obs_dim = 12
act_dim = 2
model = PPO_LSTM(obs_dim, act_dim)
# model =  PPO_Transformer_LSTM(obs_dim, act_dim)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
trainer = PPOTrainer(model, optimizer)


# Directory to save checkpoints
checkpoint_dir = 'checkpoints_head_vel'
os.makedirs(checkpoint_dir, exist_ok=True)

# Checkpoint file path
# checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_epoch_135.pth')
checkpoint_path = 'best_model.pth'

# Load checkpoint if exists
# if os.path.exists(checkpoint_path):
#     trainer.load_checkpoint(checkpoint_path)
    
env_ = FlowFieldNavEnv()
env = NormalizeObservation(env_)
# env = NormalizeReward(env1, gamma=0.995)
trainer.train(env)