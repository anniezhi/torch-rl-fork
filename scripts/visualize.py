import argparse
import numpy as np

import utils
from utils import device


# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--shift", type=int, default=0,
                    help="number of times the environment is reset at the beginning (default: 0)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="select the action with highest probability (default: False)")
parser.add_argument("--pause", type=float, default=0.1,
                    help="pause duration between two consequent actions of the agent (default: 0.1)")
parser.add_argument("--gif", type=str, default=None,
                    help="store output as gif with the given filename")
parser.add_argument("--episodes", type=int, default=1000000,
                    help="number of episodes to visualize")
parser.add_argument("--memory", action="store_true", default=False,
                    help="add a LSTM to the model")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model")
parser.add_argument("--agent-view-size", type=int, default=7,
                    help="agent vision square length")
parser.add_argument("--agent-view-type", type=str, default='self',
                    help="shuffling obstacles during episodes")
parser.add_argument("--agent-speed", type=int, default=1,
                    help="agent maximum step size at one move")
parser.add_argument("--shuffle", type=str, 
                    help="shuffling obstacles during episodes")
parser.add_argument("--random-goal", default=False, action="store_true",
                    help="randomly place the goal in the grid")
parser.add_argument("--test-mode", default=False, action="store_true",
                    help="not training, only testing code")
parser.add_argument("--no-highlight", dest='highlight', action="store_false",
                    help="highlight agent view range")
parser.set_defaults(highlight=True)

args = parser.parse_args()

# Set seed for all randomness sources

utils.seed(args.seed)

# Set device

print(f"Device: {device}\n")

# Load environment
if not args.test_mode:
    agent_model = args.model
    agent_view_size, agent_speed, seed = agent_model.split('-')[1:4]
    agent_view_size = int(agent_view_size[1:])
    agent_speed = int(agent_speed[1:])
    seed = int(seed[4:])
else:
    agent_view_size = args.agent_view_size
    agent_speed = args.agent_speed
    seed = args.seed

env = utils.make_env(args.env, seed, render_mode="human", 
                     agent_view_size=agent_view_size,
                     agent_view_type=args.agent_view_type,
                     agent_speed=agent_speed, 
                     shuffle=args.shuffle, 
                     random_goal=args.random_goal,
                     highlight=args.highlight)
                    #  rewards=[1,0])
for _ in range(args.shift):
    env.reset()
print("Environment loaded\n")

# Load agent

model_dir = utils.get_model_dir(args.model)
agent = utils.Agent(env.observation_space, env.action_space, (env.spec.kwargs['size'],env.spec.kwargs['size']), env.goal,
                    model_dir, argmax=args.argmax, use_memory=args.memory, use_text=args.text, whole_view=(args.agent_view_type=='whole'))
print("Agent loaded\n")

# Run the agent

if args.gif:
    from array2gif import write_gif

    frames = []

# Create a window to view the environment
env.render()

env_grids = []
env_goals = []

for episode in range(args.episodes):
    obs, _ = env.reset()
    env_grid = env.get_grid().encode()
    env_goal = env.get_goal()
    env_goal_mask = np.zeros_like(env_grid)
    env_goal_mask[env_goal] = 1
    env_goal = env_grid * env_goal_mask

    while True:
        env.render()
        if args.gif:
            frames.append(np.moveaxis(env.get_frame(highlight=args.highlight), 2, 0))

        action = agent.get_action(obs)
        obs, reward, terminated, truncated, _, _, _ = env.step(action)
        done = terminated | truncated
        agent.analyze_feedback(reward, done)

        if done or env.window.closed:
            if args.gif:
                print("Saving episode... ", end="")
                write_gif(np.array(frames), args.gif+str(episode)+".gif", fps=1/args.pause)
                env_grids.append(env_grid)
                env_goals.append(env_goal)
                print("Saved episode {}".format(episode))
                frames = []
            break

    if env.window.closed:
        break

if args.gif:
    env_grids = np.array(env_grids)
    print('Saving grids... ', end="")
    np.save(args.gif+'env_grids.npy', env_grids)

    env_goals = np.array(env_goals)
    print('Saving goals... ', end="")
    np.save(args.gif+'env_goals.npy', env_goals)
print('Done.')
# if args.gif:
#     print("Saving gif... ", end="")
#     write_gif(np.array(frames), args.gif+".gif", fps=1/args.pause)
#     print("Done.")
