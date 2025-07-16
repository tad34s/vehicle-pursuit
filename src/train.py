import argparse
import datetime
from pathlib import Path

import torch
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.exception import UnityActionException
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from tensorboard import program

# for TensorBoard
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

import follower_agent.hyperparameters as follower_hyperparams
import leader_agent.hyperparameters as leader_hyperparams
from agent_interface import Agent
from data_channel import DataChannel
from environment_parameters import set_parameters
from follower_agent.agent import FollowerAgent
from keyboard_listener import KeyboardListener
from leader_agent.agent import LeaderAgent
from variables import (
    MAX_TRAINED_EPISODES,
    MODEL_PATH,
    NUM_TRAINING_EXAMPLES,
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
    memory_size: int,
    agents: list[Agent],
) -> None:
    n_steps_gathered = 0
    bar = tqdm(total=memory_size)
    agents = [leader_agent, follower_agent]
    for n_steps in range(0, memory_size, NUM_AREAS):
        bar.update(NUM_AREAS)
        for agent in agents:
            steps = env.get_steps(agent.behavior_name)
            action_tuple = agent.submit_actions(steps)
            if action_tuple is not None:
                env.set_actions(agent.behavior_name, action_tuple)

        env.step()


if __name__ == "__main__":
    # Start TensorBoard
    log_location = Path(__file__).parent / "runs"
    writer = SummaryWriter(log_location / datetime.datetime.now().strftime("%y-%m-%d %H%M%S"))
    launch_tensor_board(log_location)

    # Start keyboard listener for saving
    listener = KeyboardListener()
    listener.start()

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

    leader_agent = LeaderAgent(
        visual_input_shape=leader_hyperparams.VISUAL_INPUT_SHAPE,
        nonvis_input_shape=leader_hyperparams.NONVISUAL_INPUT_SHAPE,
        buffer_size=NUM_TRAINING_EXAMPLES,
        device=device,
        num_agents=NUM_AREAS,
        writer=writer,
    )

    follower_agent = FollowerAgent(
        visual_input_shape=follower_hyperparams.VISUAL_INPUT_SHAPE,
        nonvis_input_shape=follower_hyperparams.NONVISUAL_INPUT_SHAPE,
        buffer_size=NUM_TRAINING_EXAMPLES,
        device=device,
        num_agents=NUM_AREAS,
        writer=writer,
    )

    agents = [leader_agent, follower_agent]

    # remove agent if not in env -- so we could use the same script to train only leader
    for agent in agents:
        try:
            steps = env.get_steps(agent.behavior_name)
        except UnityActionException:
            agents.remove(agent)
            continue

    try:
        if SAVE_MODEL:
            print(f"---- Will save all models to {model_folder} ----")  # noqa: T201
        else:
            print('---- Not saving model as the -s flag is default to "False" ----')  # noqa: T201

        for episode in range(MAX_TRAINED_EPISODES):
            print("------Training------")  # noqa: T201
            print(f"Episode {episode}")
            run_episode(env, NUM_TRAINING_EXAMPLES, agents)
            print("------Done------")  # noqa: T201

            for agent in agents:
                reward = agent.train()
                reward /= NUM_TRAINING_EXAMPLES
                writer.add_scalar(f"Reward/Episode {agent.name}", reward, episode)

            writer.flush()

            if SAVE_MODEL or listener.was_pressed():
                folder = Path(model_folder)
                folder.mkdir(parents=True, exist_ok=True)

                for agent in agents:
                    folder_for_agent = folder / agent.name
                    folder_for_agent.mkdir(parents=True, exist_ok=True)
                    agent.save_model(folder_for_agent / f"model-episode-{episode}.onnx")
                listener.reset()

    except KeyboardInterrupt:
        print("\nTraining interrupted")

    finally:
        env.close()
        writer.close()
