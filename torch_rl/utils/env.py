import gymnasium as gym


def make_env(env_key, seed=None, render_mode=None, **kwargs):
    if 'rewards' in kwargs.keys():
        env = gym.make(env_key, render_mode=render_mode, 
                    agent_view_size=kwargs.get('agent_view_size'),
                    agent_view_type=kwargs.get('agent_view_type'),
                    agent_speed=kwargs.get('agent_speed'), 
                    shuffle=kwargs.get('shuffle'),
                    random_goal=kwargs.get('random_goal'),
                    rewards=kwargs.get('rewards'))
    else:
        env = gym.make(env_key, render_mode=render_mode, 
                    agent_view_size=kwargs.get('agent_view_size'),
                    agent_view_type=kwargs.get('agent_view_type'),
                    agent_speed=kwargs.get('agent_speed'), 
                    shuffle=kwargs.get('shuffle'),
                    random_goal=kwargs.get('random_goal'))
    env.reset(seed=seed)
    return env
