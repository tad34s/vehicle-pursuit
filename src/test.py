import argparse
from pathlib import Path

import torch
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.exception import UnityActionException
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

import follower_agent.hyperparameters as follower_hyperparams
import leader_agent.hyperparameters as leader_hyperparams
from agent_interface import Agent
from data_channel import DataChannel
from environment_parameters import set_parameters
from follower_agent.agent import FollowerAgent
from leader_agent.agent import LeaderAgent

parser = argparse.ArgumentParser()
parser.add_argument("leader")
parser.add_argument("-f", "--follower")
parser.add_argument("-c", "--inject-correct", action="store_true")
parser.add_argument(
    "-e",
    "--env",
    default=str(Path(__file__).parent / "build" / "StandaloneLinux64" / "vehiclePursuit.x86_64"),
)


def print_env_info(env: UnityEnvironment) -> None:
    # get the action space and observation space
    behavior_names = list(env.behavior_specs.keys())
    print(f"Name of behaviors : {behavior_names}")  # noqa: T201
    spec = env.behavior_specs[behavior_names[1]]
    observation_shape = spec.observation_specs
    print(observation_shape)  # noqa: T201
    num_actions = spec.action_spec
    print(num_actions)  # noqa: T201


if __name__ == "__main__":
    # Start TensorBoard
    # Set up the environment

    args = parser.parse_args()
    follower_path = args.follower
    leader_path = args.leader
    inject_correct = args.inject_correct
    env_path = args.env

    leader_only = not follower_path

    engine_channel = EngineConfigurationChannel()
    data_channel = DataChannel()
    data_channel.set_bool_parameter("leaderOnly", leader_only)
    env_location = env_path
    env = UnityEnvironment(
        file_name=env_location,
        num_areas=1,
        side_channels=[engine_channel, data_channel],
    )

    set_parameters(data_channel)

    env.reset()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")  # noqa: T201
    print_env_info(env)

    leader_agent = Agent.from_onnyx(
        LeaderAgent,
        leader_path,
        leader_hyperparams.VISUAL_INPUT_SHAPE,
        leader_hyperparams.NONVISUAL_INPUT_SHAPE,
    )
    if follower_path:
        follower_agent = Agent.from_onnyx(
            FollowerAgent,
            follower_path,
            follower_hyperparams.VISUAL_INPUT_SHAPE,
            follower_hyperparams.NONVISUAL_INPUT_SHAPE,
            inject_correct=inject_correct,
        )

        agents = [leader_agent, follower_agent]
    else:
        agents = [leader_agent]

    # remove agent if not in env -- so we could use the same script to train only leader
    for agent in agents:
        try:
            steps = env.get_steps(agent.behavior_name)
        except UnityActionException:
            agents.remove(agent)
            continue

    try:
        while True:
            for agent in agents:
                steps = env.get_steps(agent.behavior_name)
                action_tuple = agent.submit_actions(steps)
                if action_tuple is not None:
                    env.set_actions(agent.behavior_name, action_tuple)

            env.step()

    except KeyboardInterrupt:
        print("\nTraining interrupted")

    finally:
        env.close()
