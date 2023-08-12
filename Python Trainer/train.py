import matplotlib.pyplot as plt
from mlagents_envs.environment import UnityEnvironment
from network import QNetwork
from trainer import Trainer
import os
import datetime
from variables import max_trained_epochs,exploration_chance_start,exploration_reduce,num_training_examples

if __name__ == "__main__":
	# set up the environment
	env_location = './env/Self driving.exe'
	env = UnityEnvironment(file_name=env_location, num_areas=1) # TODO: Make more num_areas work
	env.reset()

	# get the action space and observation space
	print(env.behavior_specs)
	behavior_name = list(env.behavior_specs)[0]
	print(f"Name of the behavior : {behavior_name}")
	spec = env.behavior_specs[behavior_name]
	observation_shape = spec.observation_specs
	print(observation_shape)
	num_actions = spec.action_spec
	print(num_actions)

	num_epochs = max_trained_epochs
	expl_chance = exploration_chance_start
	expl_reduce = exploration_reduce

	results = []
	try:
		qnet = QNetwork(visual_input_shape = (1, 64, 64), nonvis_input_shape=(1,1), encoding_size=126)
		trainer = Trainer(model=qnet,buffer_size=num_training_examples)

		folder_name = f"./models/{datetime.datetime.now().strftime('%d-%m-%y %H%M%S')}"
		os.makedirs(folder_name)
		print(f'---- Will save models into {folder_name}')

		for epoch in range(num_epochs):
			print(f"epoch: {epoch}, exploration chance:{expl_chance}")
			reward = trainer.train(env, expl_chance)
			results.append(reward)
			expl_chance *= expl_reduce
			trainer.save_model(f'{folder_name}/model-epoch-{epoch}.onnx')
			print(f"reward earned: {reward}")

	except KeyboardInterrupt:
		print("\nTraining interrupted, continue to next cell to save to save the model.")

	finally:
		env.close()

	# Show the training graph
	try:
		plt.plot(range(num_epochs), results)
		plt.show()
	except ValueError:
		print("\nPlot failed on interrupted training.")
