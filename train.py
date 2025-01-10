import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO  # or SAC, TD3, etc.
from stable_baselines3.common.vec_env import DummyVecEnv
from sslenv import SSLExampleEnv  # Your environment file
from gymnasium.envs.registration import register

# Import one of the available rSoccer environments
# For example, an SSL environment:
import rsoccer_gym

register(
    id="SSL-Project",
    entry_point="sslenv:SSLExampleEnv"
)


def make_env():
    env = gym.make("SSL-Project")

    return env 




if __name__ == "__main__":
    # Wrap environment in a vectorized interface
    # (often required by stable baselines)
    env = DummyVecEnv([make_env])

    # Initialize a PPO model (you can also try SAC, TD3, etc.)
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        verbose=1,
    )

    # Train the model
    # Increase total_timesteps for longer training
    model.learn(total_timesteps=30000)

    # Save model for reuse
    model.save("rSoccer_ppo_model")
    env.close()

    # ------------------------------------------------------
    # OPTIONAL: Test / Evaluate the trained policy
    # ------------------------------------------------------
    test_env = gym.make("SSL-Project", render_mode="human")  # new instance
    obs, _ = test_env.reset()
    for _ in range(1000):  # run for 1000 timesteps
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = test_env.step(action)
        print(reward)

        # Render if you like (set "render_mode" to "human" in make_env)
        # test_env.render()

        if done or truncated:
            obs, _ = test_env.reset()

    test_env.close()


