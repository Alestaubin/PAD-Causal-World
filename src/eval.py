import numpy as np
import torch
import os
from copy import deepcopy
from tqdm import tqdm
import utils
from video import VideoRecorder

from arguments import parse_args
from agent.agent import make_agent
from utils import get_curl_pos_neg


def evaluate(env, device, agent, args, video, adapt=False):
	"""Evaluate an agent, optionally adapt using PAD"""
	episode_rewards = []
	num_success = 0
	steps_to_success = []

	for i in tqdm(range(args.pad_num_episodes)):
		ep_agent = deepcopy(agent) # make a new copy

		video.init(enabled=True)

		obs = env.reset()
		done = False
		episode_reward = 0
		losses = []
		step = 0
		ep_agent.train()

		while not done:
			# Take step
			with utils.eval_mode(ep_agent):
				action = ep_agent.select_action(obs)
			next_obs, reward, done, info = env.step(action)
			episode_reward += reward
			
			success = info['success'] 
			if success:
				num_success += 1
				steps_to_success.append(step)
				break

			# Make self-supervised update if flag is true
			if adapt:				
				if args.use_inv: # inverse dynamics model

					# Prepare batch of observations
					if args.obs_type == 'pixel':
						batch_obs = utils.batch_from_obs(torch.Tensor(obs).to(device), batch_size=args.pad_batch_size)
						batch_next_obs = utils.batch_from_obs(torch.Tensor(next_obs).to(device), batch_size=args.pad_batch_size)
						# Crop if pixel observations
						batch_obs = utils.random_crop(batch_obs)
						batch_next_obs = utils.random_crop(batch_next_obs)
					
					else: # structured obs
						batch_obs = utils.batch_from_obs_structured(torch.Tensor(obs).to(device), batch_size=args.pad_batch_size)
						batch_next_obs = utils.batch_from_obs_structured(torch.Tensor(next_obs).to(device), batch_size=args.pad_batch_size)
						
						noise_std = 0.01
						batch_obs += torch.randn_like(batch_obs) * noise_std
						batch_next_obs += torch.randn_like(batch_next_obs) * noise_std
						
					batch_action = torch.Tensor(action).to(device).unsqueeze(0).repeat(args.pad_batch_size, 1)

					losses.append(ep_agent.update_inv(batch_obs, batch_next_obs, batch_action, adapt=True))

			video.record(env, losses)
			obs = next_obs
			step += 1

		video.save(f'{args.mode}_pad_{i}.mp4' if adapt else f'{args.mode}_eval_{i}.mp4')
		episode_rewards.append(episode_reward)
	
	return {
		'avg_ep_reward': np.mean(episode_rewards),
		'success_rate': num_success / args.pad_num_episodes,
		'avg_steps_to_success': np.mean(steps_to_success) if len(steps_to_success) > 0 else float('inf')
	}
	# return np.mean(episode_rewards)


def init_env(args):
		utils.set_seed_everywhere(args.seed)
		if args.domain_name == 'causalworld':
			from env.CausalWorld_wrappers import make_pad_env_causalworld
			env = make_pad_env_causalworld(
					task_name=args.task_name,
					seed=args.seed,
					episode_length=args.episode_length,
					frame_stack=args.frame_stack,
					action_repeat=args.action_repeat,
					mode=args.mode,
					camera_index=[0], 
					enable_visualization=False,
					obs_type=args.obs_type # structured or pixel 
			)
		else:
			from env.wrappers import make_pad_env
			env = make_pad_env(
				domain_name=args.domain_name,
				task_name=args.task_name,
				seed=args.seed,
				episode_length=args.episode_length,
				action_repeat=args.action_repeat,
				mode=args.mode
			)
		return env


def main(args):
	# Initialize environment
	env = init_env(args)
	model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
	video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))
	video = VideoRecorder(video_dir if args.save_video else None, height=448, width=448)
	device = torch.device("mps")#("mps" if torch.backends.mps.is_available() else "cpu")
	print("Using device:", device)
	# Prepare agent
	if args.obs_type == 'pixel':
		obs_shape = (3*args.frame_stack, 84, 84) # cropped pixel obs
	else:
		obs_shape = env.observation_space.shape
	agent = make_agent(
		obs_type=args.obs_type,
		obs_shape=obs_shape,
		device=device,
		action_shape=env.action_space.shape,
		args=args
	)
	agent.load(model_dir, args.pad_checkpoint)

	# Evaluate agent without PAD
	print(f'Evaluating {args.work_dir} for {args.pad_num_episodes} episodes (mode: {args.mode})')
	results = evaluate(env, device, agent, args, video)
	eval_reward = results['avg_ep_reward']
	print('eval reward:', int(results['avg_ep_reward']), 'success rate:', results['success_rate'], 'avg steps to success:', results['avg_steps_to_success'])
	
	# Evaluate agent with PAD (if applicable)
	pad_reward = None
	if args.use_inv or args.use_curl or args.use_rot:
		env = init_env(args)
		print(f'Policy Adaptation during Deployment of {args.work_dir} for {args.pad_num_episodes} episodes (mode: {args.mode})')
		results_pad = evaluate(env, device, agent, args, video, adapt=True)
		print('PAD eval reward:', int(results_pad['avg_ep_reward']), 'success rate:', results_pad['success_rate'], 'avg steps to success:', results_pad['avg_steps_to_success'])
		pad_reward = results_pad['avg_ep_reward']

	# Save results
	results_fp = os.path.join(args.work_dir, f'pad_{args.mode}.pt')
	torch.save({
		'args': args,
		'eval_reward': eval_reward,
		'pad_reward': pad_reward
	}, results_fp)
	print('Saved results to', results_fp)

if __name__ == '__main__':
	args = parse_args()
	main(args)
