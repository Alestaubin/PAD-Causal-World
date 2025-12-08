import numpy as np
import torch
import os
import gym
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
import matplotlib.pyplot as plt
import utils
from video import VideoRecorder

from arguments import parse_args
from agent.agent import make_agent
import seaborn as sns

def plot_average_cumulative_reward(all_step_histories, work_dir, scale):
    """
    Plots the average cumulative reward over time (steps) within an episode,
    aggregated across all test episodes.
    
    Args:
        all_step_histories (dict): Key="{method}_scale{scale}" -> Value=[[r1, r2...], [r1, r2...]...]
        work_dir (str): Save path.
        scale (float): The specific scale to plot (usually the hardest one).
    """
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    # Iterate through the 3 methods for this specific mass scale
    methods = ['no_tta', 'episodic', 'lifelong']
    colors = {'no_tta': 'gray', 'episodic': 'blue', 'lifelong': 'green'}
    
    for method in methods:
        key = f"{method}_scale{scale}"
        if key not in all_step_histories:
            continue
            
        # 1. Convert list of lists to 2D array (Episodes x Steps)
        # We assume fixed episode lengths. If lengths vary, we pad with last value or 0.
        raw_steps = all_step_histories[key]
        min_len = min(len(ep) for ep in raw_steps)
        # Truncate to min_len to be safe for numpy operations
        steps_array = np.array([ep[:min_len] for ep in raw_steps])
        
        # 2. Compute Cumulative Sum along the step axis (axis 1)
        # Shape: (Num_Episodes, Num_Steps)
        cum_rewards = np.cumsum(steps_array, axis=1)
        
        # 3. Compute Mean and Std Error across episodes (axis 0)
        mean_curve = np.mean(cum_rewards, axis=0)
        std_curve = np.std(cum_rewards, axis=0)
        stderr_curve = std_curve / np.sqrt(len(raw_steps))
        
        # 4. Plot
        x_axis = range(len(mean_curve))
        plt.plot(x_axis, mean_curve, label=method.capitalize(), color=colors[method], linewidth=2)
        plt.fill_between(x_axis, mean_curve - stderr_curve, mean_curve + stderr_curve, 
                         color=colors[method], alpha=0.15)

    plt.title(f'Average Cumulative Reward (Mass Scale x{scale})', fontsize=14)
    plt.xlabel('Episode Step')
    plt.ylabel('Cumulative Reward')
    plt.legend()
    
    save_path = os.path.join(work_dir, f'plot_cumulative_reward_scale{scale}.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Cumulative reward plot saved to {save_path}")


def plot_results(df, raw_curves, work_dir):
    """
    Generates and saves the three key analysis plots.
    """
    sns.set_theme(style="whitegrid")
    
    # 1. Robustness Curve (Avg Return vs. Mass Scale)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='Scale', y='Avg_Return', hue='Method', marker='o', linewidth=2.5)
    plt.title('Robustness to Mass Scaling (OOD Shifts)', fontsize=14)
    plt.ylabel('Average Episode Return')
    plt.xlabel('Scale Factor')
    plt.savefig(os.path.join(work_dir, 'plot_robustness_curve.png'))
    plt.close()

    # 2. Adaptation Learning Curve (Lifelong TTA)
    # We plot this for the hardest difficulty (highest mass scale) to see learning best.
    max_scale = df['Scale'].max()
    plt.figure(figsize=(10, 6))
    
    # Extract lifelong curve for max scale
    key = f"lifelong_scale{max_scale}"
    if key in raw_curves:
        data = raw_curves[key]
        # Smooth the curve for readability
        smoothed = pd.Series(data).rolling(window=5, min_periods=1).mean()
        plt.plot(data, alpha=0.3, color='green', label='Raw')
        plt.plot(smoothed, color='green', linewidth=2, label='Smoothed (MA-5)')
        plt.title(f'Lifelong Adaptation Learning Curve (Mass x{max_scale})', fontsize=14)
        plt.xlabel('Episode Index')
        plt.ylabel('Episode Return')
        plt.legend()
        plt.savefig(os.path.join(work_dir, 'plot_lifelong_learning.png'))
    plt.close()

    # 3. Intra-Episode Regret (Episodic TTA)
    # Compare Early vs Late reward to show "rapid adaptation"
    episodic_df = df[df['Method'] == 'episodic']
    if not episodic_df.empty:
        plt.figure(figsize=(10, 6))
        
        # Melt dataframe for easier plotting with seaborn
        melted = episodic_df.melt(id_vars=['Scale'], 
                                  value_vars=['Early_Episode_Reward', 'Late_Episode_Reward'], 
                                  var_name='Phase', value_name='Reward')
        
        sns.barplot(data=melted, x='Scale', y='Reward', hue='Phase', palette='viridis')
        plt.title('Intra-Episode Adaptation: Early vs. Late', fontsize=14)
        plt.ylabel('Average Step Reward')
        plt.savefig(os.path.join(work_dir, 'plot_intra_episode_regret.png'))
    plt.close()

    print(f"Plots saved to {work_dir}")

def run_tta_experiment(env, agent, args, video, tta_mode='no_tta', scale=1.0):
    """
    Runs a single experiment block (e.g., Lifelong TTA at +30% Mass).
    
    Args:
        tta_mode (str): 'episodic', 'lifelong', or 'no_tta'
        scale (float): The OOD severity (1.0 = normal, 1.5 = heavy)
    """
    device = agent.device
    num_episodes = args.pad_num_episodes
    
    # -- METRICS STORAGE --
    metrics = {
        'episode_returns': [],
        'success_rate': 0.0,
        'intra_episode_early': [], # Avg reward of first 20% steps
        'intra_episode_late': [],  # Avg reward of last 20% steps
        'step_histories': []       # Full reward curves for analysis
    }
    
    success_count = 0

    # -- AGENT SETUP --
    # For Lifelong: We copy ONCE at the start. The agent mutates across episodes.
    if tta_mode == 'lifelong':
        current_agent = deepcopy(agent)
        current_agent.train()
    # For No-TTA: We copy once, but we will NEVER call update_inv
    elif tta_mode == 'no_tta':
        current_agent = deepcopy(agent)
    
    print(f"--> Running {tta_mode} | Scale: {scale} | Episodes: {num_episodes}")

    for i in tqdm(range(num_episodes), desc=f"{tta_mode} x{scale}"):
        
        # For Episodic: We copy at the START of every episode (Resetting memory)
        if tta_mode == 'episodic':
            current_agent = deepcopy(agent)
            current_agent.train()

        # Reset Env with OOD Parameter
        if args.mode == 'finger_link_mass':
            obs = env.reset(scale=scale)
        elif args.mode == 'goal':
            obs = env.reset(goal_displacement=scale)
        else:
            raise ValueError(f"Unknown mode {args.mode} for TTA experiment.")
        
        success = False
        done = False
        episode_reward = 0
        step_rewards = []
        video.init(enabled=(i == 0)) # Record only first episode per run
        step = 0
        while not done and step < 500:
            step += 1
            # 1. Select Action 
            with utils.eval_mode(current_agent):
                action = current_agent.select_action(obs)
            
            next_obs, reward, done, info = env.step(action)
            if info['success']:
                success = True

            episode_reward += reward
            step_rewards.append(reward)

            # 2. Test-Time Adaptation Step
            if tta_mode in ['episodic', 'lifelong']:
                batch_obs = utils.batch_from_obs_structured(torch.Tensor(obs).to(device), batch_size=args.pad_batch_size)
                batch_next_obs = utils.batch_from_obs_structured(torch.Tensor(next_obs).to(device), batch_size=args.pad_batch_size)
                
                noise_std = 0.01
                batch_obs += torch.randn_like(batch_obs) * noise_std
                batch_next_obs += torch.randn_like(batch_next_obs) * noise_std

                batch_action = torch.Tensor(action).to(device).unsqueeze(0).repeat(args.pad_batch_size, 1)
    
                current_agent.update_inv(
                    batch_obs, batch_next_obs, batch_action,
                    adapt=True
                )

            # Video recording
            if video.enabled:
                video.record(env, None)

            obs = next_obs
        
        # End of Episode Processing
        if video.enabled:
            video.save(f'{args.mode}_{tta_mode}_scale{scale}.mp4')

        # -- METRICS CALCULATION --
        metrics['episode_returns'].append(episode_reward)
        metrics['step_histories'].append(step_rewards)
        
        # Intra-Episode Regret (Early vs Late)
        split_idx = max(1, len(step_rewards) // 5) # First/Last 20%
        metrics['intra_episode_early'].append(np.mean(step_rewards[:split_idx]))
        metrics['intra_episode_late'].append(np.mean(step_rewards[-split_idx:]))

        if success:
            success_count += 1
        
    metrics['success_rate'] = (success_count / num_episodes) * 100
    return metrics

# --- 3. Main Helper ---

def init_env(args):
    utils.set_seed_everywhere(args.seed)
    # Import standard wrapper logic, then wrap with ours
    from env.CausalWorld_wrappers import make_pad_env_causalworld
    
    # Create base env
    env = make_pad_env_causalworld(
            task_name=args.task_name,
            seed=args.seed,
            episode_length=args.episode_length,
            frame_stack=args.frame_stack,
            action_repeat=args.action_repeat,
            mode=args.mode,
            obs_type=args.obs_type 
    )
    return env

def main(args):
    # Setup
    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
    video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))
    print(f"Working dir: {args.work_dir}")

    video = VideoRecorder(video_dir if args.save_video else None, height=448, width=448)

    device = torch.device('mps')

    env = init_env(args)

    # Load Agent
    if args.obs_type == 'pixel':
        obs_shape = (3*args.frame_stack, 84, 84)
    else:
        obs_shape = env.observation_space.shape
        
    agent = make_agent(
        obs_type=args.obs_type,
        obs_shape=obs_shape,
        device=device,
        action_shape=env.action_space.shape,
        args=args
    )

    # Checkpoint loading
    agent.load(model_dir, args.pad_checkpoint) 

    # --- EXPERIMENT CONFIGURATION ---
    # Define the OOD shifts (+10%, +30%, +50%)
    if args.mode == 'finger_link_mass':
        scales = [1.0, 1.1, 1.3, 1.5] 
    elif args.mode == 'goal':  
        scales = [0.0, 0.01, 0.03, 0.05]
        #scales = [0.0]

    # Define TTA Modes
    tta_modes = ['no_tta', 'episodic', 'lifelong']

    # Storage for final DataFrame
    all_results = []
    raw_curves = {} 
    all_step_histories = {}

    # --- EXPERIMENT LOOP ---
    for scale in scales:
        print(f"\n=== EVALUATING SCALE: {scale} ===")
        print(f"Scale Type: {'Mass Scaling' if args.mode == 'finger_link_mass' else 'Goal Displacement'}")

        for mode in tta_modes:
            # Run the experiment
            metrics = run_tta_experiment(env, agent, args, video, tta_mode=mode, scale=scale)
            raw_curves[f"{mode}_scale{scale}"] = metrics['episode_returns']
            all_step_histories[f"{mode}_scale{scale}"] = metrics['step_histories']
            # Aggregate stats
            avg_return = np.mean(metrics['episode_returns'])
            std_return = np.std(metrics['episode_returns'])
            avg_early = np.mean(metrics['intra_episode_early'])
            avg_late = np.mean(metrics['intra_episode_late'])
            adaptation_delta = avg_late - avg_early
            
            # Save summary
            result_entry = {
                'Scale': scale,
                'Method': mode,
                'Success_Rate': metrics['success_rate'],
                'Avg_Return': avg_return,
                'Std_Return': std_return,
                'Early_Episode_Reward': avg_early,
                'Late_Episode_Reward': avg_late,
                'Adaptation_Improvement': adaptation_delta
            }
            all_results.append(result_entry)
            
            # Optional: Save detailed learning curves for plotting later
            np.save(f"{args.work_dir}/curves_{mode}_scale{scale}.npy", metrics['episode_returns'])

    # --- DISPLAY & SAVE RESULTS ---
    df = pd.DataFrame(all_results)
    print("\n\n=== FINAL RESULTS SUMMARY ===")
    print(df[['Scale', 'Method', 'Avg_Return', 'Success_Rate', 'Adaptation_Improvement']])

    # Save to CSV
    df.to_csv(os.path.join(args.work_dir, 'tta_comparison_results.csv'), index=False)
    print(f"Saved detailed results to {os.path.join(args.work_dir, 'tta_comparison_results.csv')}")
    plot_results(df, raw_curves, args.work_dir)
    plot_average_cumulative_reward(all_step_histories, args.work_dir, scale=1.0)
    plot_average_cumulative_reward(all_step_histories, args.work_dir, scale=max(scales))

if __name__ == '__main__':
    args = parse_args()
    main(args)