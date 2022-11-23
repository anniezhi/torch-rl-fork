import gymnasium as gym


def make_env(env_key, seed=None, render_mode=None, **kwargs):
    env = gym.make(env_key, render_mode=render_mode, agent_view_size=kwargs.get('agent_view_size', 7), agent_speed=kwargs.get('agent_speed', 1))
    env.reset(seed=seed)
    return env
