import numpy as np
import gym
from collections import deque
import cv2

# CausalWorld imports
from CausalWorld.causal_world.envs import CausalWorld
from CausalWorld.causal_world.task_generators import generate_task
from CausalWorld.causal_world.wrappers.curriculum_wrappers import CurriculumWrapper
import matplotlib.pyplot as plt
def make_pad_env_causalworld(
        task_name="pushing",          # e.g., 'pushing', 'stacking'
        seed=0,
        episode_length=1000,
        frame_stack=3,
        action_repeat=4,
        mode='train',
        camera_index=[0], 
        enable_visualization=False
    ):
    """
    Make environment for PAD experiments adapted for CausalWorld.
    
    Args:
        task_name (str): The ID of the CausalWorld task.
        mode (str): 'train', 'color_hard', etc.
    """
    
    # 1. Create the CausalWorld Task
    task = generate_task(task_generator_id=task_name)
    
    # 2. Initialize Environment with visual observation enabled
    env = CausalWorld(
        task=task,
        enable_visualization=enable_visualization,  # indicates if a GUI is enabled or the environment should operate in a headless mode
        seed=seed,
        action_mode="joint_positions", # Or 'end_effector_positions' depending on your policy
        observation_mode = "pixel", # Ensure pixel observations, not structured
        normalize_actions=True,
        skip_frame=action_repeat,
        max_episode_length=episode_length,
        camera_indicies=camera_index
    )

    # 3. Wrap to Extract Pixels and Convert to (C, H, W)
    # CausalWorld returns a Dict; PAD expects a standard pixel array.
    env = CausalWorldFromPixels(env, height=100, width=100)
    
    # 4. Apply Domain Randomization (The "PAD" Adaptation logic)
    env = CausalDomainWrapper(env, mode)

    # 5. Frame Stacking (Original PAD wrapper)
    env = FrameStack(env, frame_stack)

    return env


class CausalWorldFromPixels(gym.Wrapper):
    """
    Extracts image from CausalWorld dict, resizes, and permutes to (C, H, W).
    """
    def __init__(self, env, height=100, width=100):
        gym.Wrapper.__init__(self, env)
        self._height = height
        self._width = width
        
        # Define new observation space (Channel, Height, Width)
        # Assuming RGB (3 channels)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(3, height, width), dtype=np.uint8
        )

    def reset(self):
        obs = self.env.reset()
        return self._process_obs(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._process_obs(obs), reward, done, info

    def _process_obs(self, obs):
        # 1. Get the image from the specific camera
        # Note: Ensure your CausalWorld env is configured to return this camera
        #print("Obs shape:", obs.shape)
        img = obs[0]
        
        # 2. Resize if necessary
        if img.shape[0] != self._height or img.shape[1] != self._width:
             img = cv2.resize(img, (self._width, self._height), interpolation=cv2.INTER_AREA)

        # 3. Transpose from (H, W, C) -> (C, H, W) for PyTorch/PAD compatibility
        img = np.transpose(img, (2, 0, 1))
        return img


class CausalDomainWrapper(gym.Wrapper):
    """
    Replaces the 'ColorWrapper' from the original PAD code.
    Uses CausalWorld's do_intervention.
    """
    def __init__(self, env, mode):
        gym.Wrapper.__init__(self, env)
        self._mode = mode
        
    def reset(self):
        # In CausalWorld, we randomize via reset or specific randomize functions
        if 'color' in self._mode:
            self.randomize()
        if 'video' in self._mode:
            raise NotImplementedError("Video mode with greenscreen not implemented for CausalWorld.")
        # Standard reset
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def randomize(self):
        """
        Applies domain randomization specific to CausalWorld.
        """
        
        if 'color' in self._mode:
            # randomize floor_color:
            color = np.random.uniform(0.5, 1, size=3).tolist()
            success_signal, obs = self.env.do_intervention({'floor_color': color})
            # print(f"Randomized floor color to {color}, success: {success_signal}")
        if 'goal' in self._mode:
            goal_intervention_dict = self.env.sample_new_goal()
            success_signal, obs = self.env.do_intervention(goal_intervention_dict)
            print("Goal Intervention for CF env success signal", success_signal)



# --- ORIGINAL FrameStack (Unchanged) ---
class FrameStack(gym.Wrapper):
    """Stack frames as observation"""
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=env.observation_space.dtype
        )
        # self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)