
from env.CausalWorld_wrappers import make_pad_env_causalworld
import matplotlib.pyplot as plt
def main(): 
    env = make_pad_env_causalworld(
        task_name="pushing",
        seed=42,
        episode_length=1000,
        frame_stack=3,
        action_repeat=4,
        mode='color_hard',
        camera_index=[0], 
        enable_visualization=False
    )
    
    obs = env.reset()
    print("Initial observation shape:", obs.shape)
    done = False
    total_reward = 0.0
    # plt.imshow(obs)
    # plt.show()
    while not done:
        action = env.action_space.sample()  # Random action for testing
        obs, reward, done, info = env.step(action)
        total_reward += reward


    env.close()
    
    print("Episode finished with total reward:", total_reward)

if __name__ == "__main__":
    main()