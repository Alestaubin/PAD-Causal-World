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
        enable_visualization=False,
        obs_type='pixel' # 'structured' or 'pixel'
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
    print(f"Creating {task_name} CausalWorld env with the following parameters:")
    print(f"  seed: {seed}, episode_length: {episode_length}, action_repeat: {action_repeat}, obs_type: {obs_type}")

    env = CausalWorld(
        task=task,
        enable_visualization=enable_visualization,  # indicates if a GUI is enabled or the environment should operate in a headless mode
        seed=seed,
        action_mode="joint_positions", # Or 'end_effector_positions' depending on your policy
        observation_mode = obs_type, 
        normalize_actions=True,
        skip_frame=action_repeat,
        max_episode_length=episode_length-1, # off by one, unsurprisingly
        camera_indicies=camera_index
    )

    # 3. Wrap to Extract Pixels and Convert to (C, H, W)
    # CausalWorld returns a Dict; PAD expects a standard pixel array.
    if obs_type == 'pixel':
        env = CausalWorldFromPixels(env, height=100, width=100)
        
        # 4. Apply Domain Randomization (The "PAD" Adaptation logic)
        env = CausalDomainWrapper(env, mode)

        # 5. Frame Stacking (Original PAD wrapper)
        env = FrameStack(env, frame_stack)
    else:
        # For structured observations, we can still apply domain randomization
        env = CausalDomainWrapper(env, mode)

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
        img = obs[0]
        
        # If the image is float and normalized (0.0 to 1.0), scale it up
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = img * 255.0
            img = img.astype(np.uint8)
        # --- MODIFICATION END ---

        # 2. Resize if necessary
        if img.shape[0] != self._height or img.shape[1] != self._width:
             img = cv2.resize(img, (self._width, self._height), interpolation=cv2.INTER_AREA)

        # 3. Transpose from (H, W, C) -> (C, H, W)
        img = np.transpose(img, (2, 0, 1))
        return img

import numpy as np
import gym

class CausalDomainWrapper(gym.Wrapper):
    """
    Replaces the 'ColorWrapper' from the original PAD code.
    Uses CausalWorld's do_intervention to randomize physics or goals.
    """
    def __init__(self, env, mode):
        gym.Wrapper.__init__(self, env)
        self._mode = mode
        
        # This is the default mass that the model was trained with
        self.default_link_mass = 0.02 # Example default kg
        
    def reset(self, **kwargs):
        # We forward kwargs from reset() to randomize()
        # This allows you to call env.reset(mass_scale=2.0)
        obs = self.env.reset()
        goal_displacement = kwargs.pop('goal_displacement', None)
        if goal_displacement == 0.0:
            obs = self.link_mass_intervention(fixed_mass=0.02, mass_noise=0.0)
            return self.goal_intervention(goal_displacement=0.0)
        if self._mode == 'goal':
            obs = self.link_mass_intervention(
                mass_scale=None,
                fixed_mass=0.02,
                mass_noise=0.0
            )
            return self.goal_intervention(goal_displacement=goal_displacement)
        elif self._mode == 'finger_link_mass':
            return self.link_mass_intervention(
                mass_scale=kwargs.get('mass_scale', None),
                fixed_mass=kwargs.get('fixed_mass', None),
                mass_noise=kwargs.get('mass_noise', 0.001)
            )
        elif self._mode == 'train':
            obs = self.link_mass_intervention(fixed_mass=0.02, mass_noise=0.0)
            return self.goal_intervention(goal_displacement=0.0)
        else:
            raise NotImplementedError(f"Randomization mode {self._mode} not implemented for CausalWorld.")
        
    def step(self, action):
        return self.env.step(action)
    
    def goal_intervention(self, goal_displacement):
        # cylindrical coordinates: [radius, angle, height] 
        # bounds: [[0.0, - math.pi, 0.0075], [0.11, math.pi, 0.15]]

        default_goal_60 = [0.0, 0.0, 0.1]
        default_goal_120 = [0.0, 0.0, 0.13]
        default_goal_300 = [0.0, 0.0, 0.15]
        default_goals = [default_goal_60, default_goal_120, default_goal_300]
        names = ['goal_60', 'goal_120', 'goal_300']
        
        angle = np.random.uniform(-np.pi, np.pi)
        radius = goal_displacement

        target_goals = []
        if radius == 0.0:
            # No displacement; use default goals
            target_goals = default_goals
        # else:
        #     target_goals = [[radius, angle, default_goal[2]] for default_goal in default_goals]
        else:
            for default_goal in default_goals:
                # print("Default goal:", default_goal)
                if goal_displacement is not None:
                    # Randomly sample a new goal position within a circle of radius goal_displacement
                    angle = np.random.uniform(-np.pi, np.pi)
                    radius = goal_displacement
                    target_pos = [
                        radius,
                        angle,
                        default_goal[2]
                    ]
                else:
                    raise ValueError("goal_displacement must be provided for goal randomization.")
                target_goals.append(target_pos)
        
        goal_intervention_dict = {
            names[i]: {'cylindrical_position': np.array(target_goals[i])} 
            for i in range(len(names))
        }
        # print("Goal intervention dict:", goal_intervention_dict)
        success_signal, obs = self.env.do_intervention(goal_intervention_dict)
        # print(f"Goal Intervention applied. Success={success_signal}")
        return obs
    
    def link_mass_intervention(self, mass_scale=None, fixed_mass=None, mass_noise=0.001):
        # Generate list of all finger links
        # TriFinger has 3 fingers (0, 120, 240/300 degrees), each with links 0, 1, 2
        names = (
            ['robot_finger_60_link_'+str(i) for i in range(0,3)] + 
            ['robot_finger_120_link_'+str(i) for i in range(0,3)] + 
            ['robot_finger_300_link_'+str(i) for i in range(0,3)]
        )
        
        for name in names:
            # 1. Determine Base Mass
            if fixed_mass is not None:
                target_mass = fixed_mass
            elif mass_scale is not None:
                # Scale based on the default reference
                target_mass = self.default_link_mass * mass_scale
            else:
                # Fallback to the original random uniform logic if no args provided
                target_mass = np.random.uniform(0.015, 0.045)

            # 2. Add Noise (target +/- noise)
            if mass_noise > 0:
                noise = np.random.uniform(-mass_noise, mass_noise)
                target_mass += noise

            # 3. Apply Safety Bounds ()
            target_mass = max(0.015, target_mass)
            target_mass = min(0.045, target_mass)  

            # Apply intervention
            success_signal, obs = self.env.do_intervention({
                name: {'mass': np.array([target_mass])}
            })
            # print(f"Intervention {name}: Mass={target_mass:.4f}, Success={success_signal}")
        return obs


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
            high=255,  # CHANGED: was high=1, now high=255 for uint8
            shape=((shp[0] * k,) + shp[1:]),
            dtype=np.uint8 # Explicitly set to uint8
        )

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
    

# class CausalDomainWrapper(gym.Wrapper):
#     """
#     Replaces the 'ColorWrapper' from the original PAD code.
#     Uses CausalWorld's do_intervention.
#     """
#     def __init__(self, env, mode):
#         gym.Wrapper.__init__(self, env)
#         self._mode = mode
#     def reset(self, mode, **kwargs):
#         # In CausalWorld, we randomize via reset or specific randomize functions
#         obs = self.env.reset()
#         if self._mode != 'train':
#             self.randomize(mode=self._mode, **kwargs)
#         return obs
#         # return self.env.reset()

#     def step(self, action):
#         return self.env.step(action)

#     def randomize(self, **kwargs):
#         """
#         Applies domain randomization specific to CausalWorld.
#         """
#         if 'goal' in self._mode:
#             goal_intervention_dict = self.env.sample_new_goal()
#             success_signal, obs = self.env.do_intervention(goal_intervention_dict)
#             # print("Goal Intervention for CF env success signal", success_signal)
#         elif 'finger_link_mass' in self._mode:
#             names = ['robot_finger_60_link_'+str(i) for i in range(0,3)] + ['robot_finger_120_link_'+str(i) for i in range(0,3)] + ['robot_finger_300_link_'+str(i) for i in range(0,3)]
#         for name in names:
#             mass = np.random.uniform(0.015, 0.045, [1,]) # space a
#             success_signal, obs = self.env.do_intervention({name: {'mass': mass}})
#             print("Finger link Mass Intervention for CF env success signal", success_signal)
#         elif 'mass' in self._mode:
#             # Randomize object weights
#             mass = np.random.uniform(0.015, 0.045, [1,]) # space a
#             success_signal, obs = self.env.do_intervention({'tool_block': {'mass': mass}})
#             print("Mass Intervention for CF env success signal", success_signal)
#         else:
#             raise NotImplementedError(f"Randomization mode {self._mode} not implemented for CausalWorld.")
        
#         return obs
