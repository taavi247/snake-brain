import threading
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

class SnakeNetwork:
	def __init__(self):
		print('SnakeNetwork init')
		self.network_thread = threading.Thread(
			target=self.running, args=(1,), daemon=True)

		self.env_rows = 30
		self.env_columns = 30
		self.n_actions = 4
		self.batch_size = 32
		self.hidden_size = 20
		self.gamma=0.99

		self.agent = SnakeAgent(self.env_rows * self.env_columns,
								self.n_actions, self.batch_size,
								self.hidden_size, self.gamma)

	def start_snakenetwork(self):
		self.agent.sample_database()
		#self.network_thread.start()
		print('Started network thread ' + str(self.network_thread.ident))

	def running(self, name):
		pass

	def get_action(self):
		return random.randint(0, 3)


class SnakeDQN(nn.Module):
	def __init__(self, input_dim, action_dim, hidden_size=20):
		super().__init__()
		self.linear1 = nn.Linear(input_dim, hidden_size)
		self.linear2 = nn.Linear(hidden_size, action_dim)

	def forward(self, x):
		x = F.relu(self.linear1(x))
		x = self.linear2(x)
		return x

class SnakeAgent(object):
	def __init__(self, input_dim, action_dim, batch_size=32, 
				 hidden_size=20, gamma=0.99):
		self.input_dim = input_dim

		self.policy_network = SnakeDQN(input_dim, action_dim, hidden_size)
		self.target_network = SnakeDQN(input_dim, action_dim, hidden_size)

		self.optimizer = torch.optim.Adam(
			self.policy_network.parameters(), lr=5e-4)

	def sample_database(self):
		from .models import SnakeState

		pks = SnakeState.objects.values_list('pk', flat=True)
		random_object = SnakeState.objects.get(pk=random.choice(pks))
		
		current_state = np.zeros(self.input_dim)
		current_state[random_object.apples] = 1
		current_state[random_object.scissors] = 2
		current_state[random_object.walls] = 3
		current_state[random_object.snakehead] = 4
		current_state[random_object.snakebody] = 5

		action = random_object.action

		next_state = np.zeros(self.input_dim)
		next_state_object = SnakeState.objects.filter(
			game_id=random_object.game_id,
			order_id=random_object.order_id + 1)

		reward = 0
		done = True

		if (next_state_object):
			next_state[next_state_object[0].apples] = 1
			next_state[next_state_object[0].scissors] = 2
			next_state[next_state_object[0].walls] = 3
			next_state[next_state_object[0].snakehead] = 4
			next_state[next_state_object[0].snakebody] = 5
			reward = next_state_object[0].score - random_object.score
			done = False

		return current_state, action, next_state, reward, done