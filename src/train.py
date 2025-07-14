import argparse
import datetime
from pathlib import Path

import torch
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from tensorboard import program

# for TensorBoard
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

import follower_agent.hyperparameters as follower_hyperparams
import leader_agent.hyperparameters as leader_hyperparams
from data_channel import DataChannel
from environment_parameters import set_parameters
from follower_agent.agent import FollowerAgent
from leader_agent.agent import LeaderAgent
from variables import (
    EPISODE_LENGTH,
    MAX_TRAINED_EPISODES,
    MODEL_PATH,
)

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--num-areas", type=int, default=1)
parser.add_argument("-s", "--save-model", action="store_true")
parser.add_argument(
    "-e",
    "--env",
    default=str(Path(__file__).parent / "build" / "vehiclePursuit.x86_64"),
)
parser.add_argument("-D", "--no-display", action="store_true")
parser.add_argument("-t", "--time-scale", type=float, default=1.0)
parser.add_argument("-i", "--interactive", action="store_true")
args = parser.parse_args()
NUM_AREAS = args.num_areas
SAVE_MODEL = args.save_model
ENV_PATH = args.env
NO_DISPLAY = args.no_display
TIME_SCALE = args.time_scale
INTERACTIVE = args.interactive


def launch_tensor_board(logs_location: Path) -> None:
    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", str(logs_location)])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")


def print_env_info(env: UnityEnvironment) -> None:
    # get the action space and observation space
    behavior_names = list(env.behavior_specs.keys())
    print(f"Name of behaviors : {behavior_names}")  # noqa: T201
    spec = env.behavior_specs[behavior_names[1]]
    observation_shape = spec.observation_specs
    print(observation_shape)  # noqa: T201
    num_actions = spec.action_spec
    print(num_actions)  # noqa: T201


def run_episode(
    env: UnityEnvironment,
    episode_len: int,
    leader_agent: LeaderAgent,
    follower_agent: FollowerAgent,
) -> None:
    n_steps = 0

    agents = [leader_agent, follower_agent]
    for n_steps in tqdm(range(episode_len)):
        for agent in agents:
            action_tuple = agent.submit_actions(env.get_steps(agent.behavior_name))
            if action_tuple is not None:
                env.set_actions(agent.behavior_name, action_tuple)

        env.step()


if __name__ == "__main__":
    # Start TensorBoard
    log_location = Path(__file__).parent / "runs"
    writer = SummaryWriter(log_location / datetime.datetime.now().strftime("%y-%m-%d %H%M%S"))
    launch_tensor_board(log_location)

    # Start keyboard listener for saving

    # Set up the environment
    engine_channel = EngineConfigurationChannel()
    data_channel = DataChannel()
    env_location = ENV_PATH
    if INTERACTIVE:
        env_location = None
        print("Waiting for unity environment")  # noqa: T201
    env = UnityEnvironment(
        file_name=env_location,
        num_areas=NUM_AREAS,
        side_channels=[engine_channel, data_channel],
    )

    set_parameters(data_channel)

    engine_channel.set_configuration_parameters(time_scale=TIME_SCALE)
    env.reset()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")  # noqa: T201
    print_env_info(env)

    model_folder = Path(MODEL_PATH) / datetime.datetime.now().strftime("%y-%m-%d %H%M%S")

    try:
        leader_agent = LeaderAgent(
            visual_input_shape=leader_hyperparams.VISUAL_INPUT_SHAPE,
            nonvis_input_shape=leader_hyperparams.NONVISUAL_INPUT_SHAPE,
            buffer_size=EPISODE_LENGTH,
            device=device,
            num_agents=NUM_AREAS,
            writer=writer,
        )

        follower_agent = FollowerAgent(
            visual_input_shape=follower_hyperparams.VISUAL_INPUT_SHAPE,
            nonvis_input_shape=follower_hyperparams.NONVISUAL_INPUT_SHAPE,
            buffer_size=EPISODE_LENGTH,
            device=device,
            num_agents=NUM_AREAS,
            writer=writer,
        )

        if SAVE_MODEL:
            print(f"---- Will save all models to {model_folder} ----")  # noqa: T201
        else:
            print('---- Not saving model as the -s flag is default to "False" ----')  # noqa: T201

        for episode in range(MAX_TRAINED_EPISODES):
            print("------Training------")  # noqa: T201
            print(f"Episode {episode}")
            run_episode(env, EPISODE_LENGTH, leader_agent, follower_agent)
            rewards_leader = leader_agent.train()
            rewards_follower = follower_agent.train()
            print("------Done------")  # noqa: T201

            rewards_leader /= EPISODE_LENGTH
            rewards_follower /= EPISODE_LENGTH
            writer.add_scalar("Reward/Episode Leader", rewards_leader, episode)
            writer.add_scalar("Reward/Episode Follower", rewards_follower, episode)
            writer.flush()

    except KeyboardInterrupt:
        print("\nTraining interrupted")

    finally:
        env.close()
        writer.close()
