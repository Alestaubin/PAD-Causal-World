import torch
import os

from arguments import parse_args
# from env.wrappers import make_pad_env
# from env.CausalWorld_wrappers import make_pad_env_causalworld
from agent.agent import make_agent
import utils
import time
from logger import Logger
from video import VideoRecorder
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
import numpy as np

# Configure logging

def evaluate(env, agent, video, num_episodes, L, step):
	"""Evaluate agent"""
	for i in range(num_episodes):
		obs = env.reset()
		video.init(enabled=(i == 0))
		done = False
		episode_reward = 0
		while not done:
			with utils.eval_mode(agent):
				action = agent.select_action(obs)
			obs, reward, done, _ = env.step(action)
			video.record(env)
			episode_reward += reward

		video.save('%d.mp4' % step)
		L.log('eval/episode_reward', episode_reward, step)
	L.dump(step)
	return episode_reward


def main(args):
	# write a readme file with args.note in work_dir
	if args.note is not None:
		if not os.path.exists(args.work_dir):
			os.makedirs(args.work_dir)
		note_file = os.path.join(args.work_dir, 'README.txt')
		with open(note_file, 'w') as f:
			f.write(args.note)
			f.write('\n')
			f.write(str(args))
	# Initialize environment
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

	utils.make_dir(args.work_dir)
	model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
	video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))
	video = VideoRecorder(video_dir if args.save_video else None)
	device = torch.device("mps")#("mps" if torch.backends.mps.is_available() else "cpu")
	print("Using device:", device)
	# Prepare agent
	#assert torch.cuda.is_available(), 'must have cuda enabled'
	replay_buffer = utils.ReplayBuffer(
		device=device,
		obs_shape=env.observation_space.shape,
		action_shape=env.action_space.shape,
		capacity=args.train_steps,
		batch_size=args.batch_size
	)
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

	L = Logger(args.work_dir, use_tb=False)
	episode, episode_reward, done = 0, 0, True
	start_time = time.time()
	best_eval_reward = -float('inf')
	for step in range(args.train_steps+1):
		if done:
			if step > 0:
				L.log('train/duration', time.time() - start_time, step)
				start_time = time.time()
				L.dump(step)

			# Evaluate agent periodically
			if step % args.eval_freq == 0:
				print('Evaluating:', args.work_dir)
				L.log('eval/episode', episode, step)
				eval_episode_reward = evaluate(env, agent, video, args.eval_episodes, L, step)
				
				if eval_episode_reward > best_eval_reward:
					best_eval_reward = eval_episode_reward
					if args.save_model:
						agent.save(model_dir, 'best')
				
			# Save agent periodically
			if step % args.save_freq == 0 and step > 0:
				if args.save_model:
					agent.save(model_dir, step)

			L.log('train/episode_reward', episode_reward, step)

			obs = env.reset()
			done = False
			episode_reward = 0
			episode_step = 0
			episode += 1

			L.log('train/episode', episode, step)
		if step % 1000 == 0 and args.obs_type == 'pixel':
			
			# Get the last observation from the replay buffer
			# Shape is likely (9, 100, 100) due to FrameStack=3 * RGB=3
			debug_obs = replay_buffer.obses[replay_buffer.idx - 1]
			
			# Take just the first 3 channels (the most recent frame)
			debug_img = debug_obs[:3, :, :]
			
			# Transpose from (C, H, W) back to (H, W, C) for Matplotlib
			for i in range(3):
				debug_img = np.transpose(debug_obs[i*3:(i+1)*3, :, :], (1, 2, 0))
			
				# print(f"DEBUG: Obs Min: {debug_img.min()}, Max: {debug_img.max()}, Type: {debug_img.dtype}")
				
				plt.imshow(debug_img)
				plt.savefig(f"debug_agent_view_{step}_{i}.png")
				plt.close()
		# Sample action for data collection
		if step < args.init_steps:
			action = env.action_space.sample()
		else:
			with utils.eval_mode(agent):
				action = agent.sample_action(obs)
		
		# print("observation shape:", obs.shape)
		# print("observation dtype:", obs.dtype)
		# Run training update
		if step >= args.init_steps:
			num_updates = args.init_steps if step == args.init_steps else 1
			if num_updates > 1:
				for _ in tqdm(range(num_updates), desc="Updating agent"):
					# print("updating agent")
					agent.update(replay_buffer, L, step)
			else:
				agent.update(replay_buffer, L, step)
		# Take step
		next_obs, reward, done, _ = env.step(action)
		# done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(done)
		done_bool = 0 if step % args.eval_freq == 0 else float(done)
		replay_buffer.add(obs, action, reward, next_obs, done_bool)
		episode_reward += reward
		obs = next_obs
		episode_step += 1
		prev_step = step

if __name__ == '__main__':
	args = parse_args()
	main(args)
