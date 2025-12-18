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
import time 

def get_time_str():
    return time.strftime("%Y%m%d-%H%M%S")

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
    methods = ['no_inv', 'no_tta', 'episodic', 'lifelong']
    colors = {'no_inv': 'red', 'no_tta': 'gray', 'episodic': 'blue', 'lifelong': 'green'}
    
    for method in methods:
        key = f"{method}_scale{scale}"
        if key not in all_step_histories:
            print(f"Warning: No data for {key}, skipping cumulative reward plot.")
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

    plt.title(f'Average Cumulative Reward (radius {scale})', fontsize=14)
    plt.xlabel('Episode Step')
    plt.ylabel('Cumulative Reward')
    plt.legend()
    
    save_path = os.path.join(work_dir, f'plot_cumulative_reward_radius{scale}.png')
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
    plt.title('Robustness to Goal Displacement (OOD Shifts)', fontsize=14)
    plt.ylabel('Average Episode Return')
    plt.xlabel('radius (meters)')
    plt.savefig(os.path.join(work_dir, 'plot_robustness_curve.png'))
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

def run_all_tta_experiment(env, agent, no_inv_agent, args, video, scale=1.0):
    """
    Runs an experiment block where No-TTA, Episodic, and Lifelong agents
    are evaluated on IDENTICAL environment instances for fairness.
    
    Args:
        scale (float): The OOD severity (mass or goal displacement).
    """

    device = agent.device
    num_episodes = args.pad_num_episodes

    no_inv_agent = deepcopy(no_inv_agent)

    agent_no_tta = deepcopy(agent)
    
    agent_lifelong = deepcopy(agent)
    agent_lifelong.train()
    
    # -- METRICS STORAGE --
    modes = ['no_tta', 'episodic', 'lifelong', 'no_inv']
    results = {
        mode: {
            'episode_returns': [],
            'success_count': 0,
            'step_histories': []
        } for mode in modes
    }
    
    print(f"--> Running Comparison | Scale: {scale} | Episodes: {num_episodes}")
    # generate a random seed
    random_seed = np.random.randint(0, 1000000)
    # Loop through episodes (Outer loop)
    for i in tqdm(range(num_episodes), desc=f"Eval x{scale}"):
        #episode_seed = args.seed + i
        episode_seed = random_seed

        for mode in modes:
            # print("Mode:", mode)
            # this way all modes get the same environment instances
            utils.set_seed_everywhere(episode_seed)
            
            if args.mode == 'finger_link_mass':
                obs = env.reset(scale=scale)
            elif args.mode == 'goal':
                obs = env.reset(goal_displacement=scale)
            else:
                raise ValueError(f"Unknown mode {args.mode}")

            if mode == 'no_tta':
                current_agent = agent_no_tta
                adapt = False
            elif mode == 'episodic':
                # Reset agent to original baseline state
                current_agent = deepcopy(agent) 
                current_agent.train()
                adapt = True
            elif mode == 'lifelong':
                # Continue using the persistent agent
                current_agent = agent_lifelong
                adapt = True
            elif mode == 'no_inv':
                current_agent = no_inv_agent
                adapt = False

            done = False
            episode_reward = 0
            step_rewards = []
            success = False
            
            # Init video only for the last episode of the batch
            if i == num_episodes - 1: 
                video.init(enabled=True)
                write_video = True
            else:
                write_video = False
            
            step = 0
            while not done and step < 500:
                step += 1
                
                with utils.eval_mode(current_agent):
                    action = current_agent.select_action(obs)
                
                next_obs, reward, done, info = env.step(action)
                if info.get('success', False):
                    success = True

                episode_reward += reward
                step_rewards.append(reward)

                # Test-Time Adaptation Update
                if adapt:
                    # Prepare batch (replicating single obs for batch compatibility)
                    batch_obs = torch.Tensor(obs).to(device).unsqueeze(0).repeat(args.pad_batch_size, 1)
                    batch_next_obs = torch.Tensor(next_obs).to(device).unsqueeze(0).repeat(args.pad_batch_size, 1)
                    batch_action = torch.Tensor(action).to(device).unsqueeze(0).repeat(args.pad_batch_size, 1)
                    
                    # Add noise to batch (otherwise it's just the same obs repeated) 
                    noise_std = 0.0075
                    batch_obs += torch.randn_like(batch_obs) * noise_std
                    batch_next_obs += torch.randn_like(batch_next_obs) * noise_std

                    current_agent.update_inv(
                        batch_obs, batch_next_obs, batch_action,
                        adapt=True
                    )

                if write_video:
                    video.record(env, None)

                obs = next_obs
            
            # Save Video
            if write_video:
                video.save(f'{args.mode}_{mode}_scale{scale}.mp4')
            # 5. Store Metrics
            res = results[mode]
            res['episode_returns'].append(episode_reward)
            res['step_histories'].append(step_rewards)
            if success:
                res['success_count'] += 1

    # Calculate final aggregate stats for all modes
    final_metrics = {}
    for mode in modes:
        res = results[mode]
        final_metrics[mode] = {
            'episode_returns': res['episode_returns'],
            'step_histories': res['step_histories'],
            'success_rate': (res['success_count'] / num_episodes) * 100,
        }
    return final_metrics

def main(args):

    work_dir = args.work_dir+"/"+get_time_str()
    # Setup
    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
    video_dir = utils.make_dir(os.path.join(work_dir, 'video'))
    print(f"Working dir: {work_dir}")
    video = VideoRecorder(video_dir if args.save_video else None, height=448, width=448)

    device = torch.device('mps')

    env = init_env(args)

    obs_shape = env.observation_space.shape
    
    args_no_inv = deepcopy(args) 
    args_no_inv.use_inv = False
    args_no_inv.work_dir = "logs/causal_world_reaching/no_inv/2/model"
    no_inv_agent =  make_agent(
		obs_type=args_no_inv.obs_type,
		obs_shape=obs_shape,
		device=device,
		action_shape=env.action_space.shape,
		args=args_no_inv
	)
    no_inv_agent.load(args_no_inv.work_dir, 'best')

    agent = make_agent(
        obs_type=args.obs_type,
        obs_shape=obs_shape,
        device=device,
        action_shape=env.action_space.shape,
        args=args
    )
    agent.load(model_dir, args.pad_checkpoint) 

    # --- EXPERIMENT CONFIGURATION ---
    # Define the OOD shifts (+10%, +30%, +50%)
    if args.mode == 'finger_link_mass':
        scales = [2.0]
    elif args.mode == 'goal':
        scales = [0.0, 0.01, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045]

    # Storage for final DataFrame
    all_results = []
    raw_curves = {}
    all_step_histories = {}

    # --- EXPERIMENT LOOP ---
    for scale in scales:
        print(f"\n=== EVALUATING SCALE: {scale} ===")
        
        # Run all 3 modes (no tta, episodic, lifelong)
        metrics_all_modes = run_all_tta_experiment(env, agent=agent, no_inv_agent=no_inv_agent, args=args, video=video, scale=scale)

        # Process results for each mode
        for mode, metrics in metrics_all_modes.items():
            
            # Save raw data for plots
            raw_curves[f"{mode}_scale{scale}"] = metrics['episode_returns']
            all_step_histories[f"{mode}_scale{scale}"] = metrics['step_histories']
            
            # Aggregate
            avg_return = np.mean(metrics['episode_returns'])
            std_return = np.std(metrics['episode_returns'])
            
            result_entry = {
                'Scale': scale,
                'Method': mode,
                'Success_Rate': metrics['success_rate'],
                'Avg_Return': avg_return,
                'Std_Return': std_return,
            }
            print(f"Method: {mode} | Scale: {scale} | Avg Return: {avg_return:.2f} | Success Rate: {metrics['success_rate']:.2f}%")
            all_results.append(result_entry)

            # plot reward per episode curve for lifelong at this scale
            if mode == 'lifelong':
                plt.figure(figsize=(10, 6))
                data = metrics['episode_returns']
                smoothed = pd.Series(data).rolling(window=5, min_periods=1).mean()
                plt.plot(data, alpha=0.3, color='green', label='Raw')
                plt.plot(smoothed, color='green', linewidth=2, label='Smoothed (MA-5)')
                plt.title(f'Lifelong Adaptation Learning Curve (Radius {scale})', fontsize=14)
                plt.xlabel('Episode Index')
                plt.ylabel('Episode Return')
                plt.legend()
                plt.savefig(os.path.join(work_dir, f'plot_lifelong_learning_radius{scale}.png'))
                plt.close()
    print("all_step_histories keys:", all_step_histories.keys())
    # Save and Plot
    df = pd.DataFrame(all_results)
    print("\n=== FINAL RESULTS SUMMARY ===")
    print(df[['Scale', 'Method', 'Avg_Return', 'Std_Return', 'Success_Rate']])

    df.to_csv(os.path.join(work_dir, 'tta_comparison_results.csv'), index=False)
    
    # Ensure you have your plotting functions defined or imported
    plot_results(df, raw_curves, work_dir)
    # plot_average_cumulative_reward(all_step_histories, work_dir, scale=1.0)
    # plot_average_cumulative_reward(all_step_histories, work_dir, scale=max(scales))

if __name__ == '__main__':
    args = parse_args()
    main(args)