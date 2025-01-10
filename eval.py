from stable_baselines3 import PPO
from sslenv import SSLExampleEnv  # Your environment file

# 1) Create your custom environment in "human" mode
env = SSLExampleEnv(render_mode="human")

# 2) Load the trained model
model = PPO.load("rSoccer_ppo_model", env=env)

# 3) Reset environment
obs, info = env.reset()

# 4) Loop and render
for _ in range(100000):  # or whichever length
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)

    # Visualize
    env.render()

    if done or truncated:
        obs, info = env.reset()

env.close()
