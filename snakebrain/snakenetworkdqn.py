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
		self.thread_run = True
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
		self.network_thread.start()
		print('Started network thread ' + str(self.network_thread.ident))

	def running(self, name):
		kl = 0
		kt = 0
		loss = 0
		k_loss = 5000
		k_target_update = 10000

		while (self.thread_run):
			loss += self.agent.update_network()
			kl += 1
			kt += 1
			if (kl > k_loss):
				print(loss / k_loss)
				loss = 0
				kl = 0
			if (kt > k_target_update):
				self.agent.update_target_network()
				torch.save(
					self.agent.policy_network.state_dict(),
					"policy_weights.mdl")
				kt = 0
		
	def get_action(self, current_state, epsilon=0.05):
		explore = random.random()
		if explore > epsilon:
			with torch.no_grad():
				current_state = torch.tensor(current_state, dtype=torch.float32)
				q_values = self.agent.policy_network(current_state)
				return torch.argmax(q_values).item()
		else:
			return random.randint(0, 3)

class SnakeNN(nn.Module):
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
		self.action_dim = action_dim
		self.batch_size = batch_size
		self.gamma = gamma

		self.policy_network = SnakeNN(input_dim, action_dim, hidden_size)
		self.policy_network.load_state_dict(torch.load("policy_weights.mdl"))
		self.target_network = SnakeNN(input_dim, action_dim, hidden_size)
		self.target_network.load_state_dict(self.policy_network.state_dict())
		self.target_network.eval()

		self.optimizer = torch.optim.Adam(
			self.policy_network.parameters(), lr=5e-4)

	def update_network(self):
		current_state, action, next_state, reward, done = self.sample_database()

		current_state = torch.tensor(current_state, dtype=torch.float32)
		current_action = torch.tensor(action, dtype=torch.int64)
		next_state = torch.tensor(next_state, dtype=torch.float32)
		reward = torch.tensor(reward, dtype=torch.uint8)
		done_mask = torch.tensor(done, dtype=torch.bool)

		q_values = self.policy_network(current_state).gather(-1, current_action)

		next_q_values = torch.zeros(1)
		next_q_values = self.target_network(current_state * ~done_mask).max()

		expected_q_values = reward + self.gamma * next_q_values

		loss = F.smooth_l1_loss(q_values, expected_q_values)

		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		return loss.item()

	def update_target_network(self):
		self.target_network.load_state_dict(self.policy_network.state_dict())

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