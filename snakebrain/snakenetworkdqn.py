import threading
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from os.path import exists

class SnakeNetwork:
	def __init__(self):
		print('SnakeNetwork Init')
		self.thread_run = True
		self.network_thread = threading.Thread(
			target=self.running, args=(1,), daemon=True
		)

		self.input_dim = 20 * 20
		self.n_actions = 4
		self.batch_size = 32
		self.hidden_size = 20

		# Future rewards discount
		# Lower the number more the agent prefers actions
		# that maximize rewards in its near future
		self.gamma = 0.99

		# Epsilon 1.0: The agent chooses action 100% randomly
		# Epsilon 0.0: The agent always uses its policy to get action
		self.epsilon = 0.05 

		self.agent = SnakeAgent(self.input_dim, self.n_actions, 
								self.batch_size, self.hidden_size,
								self.gamma
		)

	def start_snakenetwork(self):
		from .models import SnakeState

		if (SnakeState.objects.count() > 500):
			self.network_thread.start()
			print('Started network thread ' + str(self.network_thread.ident))
		else:
			print("Not enough samples in database to begin learning!")
			self.epsilon = 1.01

	def running(self, name):
		k = 0
		loss = 0
		loss_best = float('inf')
		k_target_update = 5000

		while (self.thread_run):
			loss += self.agent.update_network()
			k += 1
			if (k > k_target_update):
				loss_avg = loss / k_target_update
				print('Loss: ', loss_avg)

				self.agent.update_target_network()

				if (loss_best > loss_avg):
					loss_best = loss_avg
					torch.save(
						self.agent.policy_network.state_dict(),
						"snakebrain/policy_weights_best.mdl")
				else:
					torch.save(
						self.agent.policy_network.state_dict(),
						"snakebrain/policy_weights.mdl")

				loss = 0
				k = 0

			# Signal to let other threads run
			time.sleep(0)
		
	def get_action(self, snakestate):
		from .models import SnakeState

		current_object = snakestate
		
		current_state = np.zeros(self.input_dim)
		current_state[current_object.apples] = 0.2
		current_state[current_object.scissors] = 0.4
		current_state[current_object.walls] = 0.6
		current_state[current_object.snakehead] = 0.8
		current_state[current_object.snakebody] = 1.0
		
		action = None
		
		# Choose if to explore environment by choosing random action
		# or use the agent policy to get the optimal action
		explore = random.random()
		if explore > self.epsilon:
			with torch.no_grad():
				current_state = torch.tensor(current_state, dtype=torch.float32)
				q_values = self.agent.policy_network(current_state)
				action = torch.argmax(q_values).item()
		else:
			action = random.randint(0, 3)
		
		current_object.action = action

		# Saves the current object and chosen action
		# in the model database
		current_object.save()

		return action

# Configure what kind of neural network the agents uses here
class SnakeNN(nn.Module):
	def __init__(self, input_dim, action_dim, hidden_size=20):
		super().__init__()
		self.linear1 = nn.Linear(input_dim, hidden_size)
		self.linear2 = nn.Linear(hidden_size, hidden_size)
		self.linear3 = nn.Linear(hidden_size, action_dim)

	def forward(self, x):
		x = F.relu(self.linear1(x))
		x = F.relu(self.linear2(x))
		return self.linear3(x)

class SnakeAgent(object):
	def __init__(self, input_dim, action_dim, batch_size=32, 
				 hidden_size=20, gamma=0.99):
		self.input_dim = input_dim
		self.action_dim = action_dim
		self.batch_size = batch_size
		self.gamma = gamma

		self.policy_network = SnakeNN(input_dim, action_dim, hidden_size)

		if (exists('snakebrain/policy_weights_best.mdl')):
			self.policy_network.load_state_dict(
				torch.load("snakebrain/policy_weights_best.mdl")
		)
		
		self.target_network = SnakeNN(input_dim, action_dim, hidden_size)
		self.target_network.load_state_dict(self.policy_network.state_dict())
		self.target_network.eval()

		self.optimizer = torch.optim.Adam(
			self.policy_network.parameters(), lr=3e-4)

	def update_network(self):
		current_state, action, next_state, reward, done = self.sample_database()

		current_state = torch.tensor(current_state, dtype=torch.float32)
		current_action = torch.tensor(action, dtype=torch.int64)
		next_state = torch.tensor(next_state, dtype=torch.float32)
		reward = torch.tensor(reward, dtype=torch.uint8)
		done_mask = torch.tensor(done, dtype=torch.bool)

		q_values = self.policy_network(current_state).gather(-1, current_action)

		# If next state is final (The snake is fed or dead) then
		# use Q-value of 0 for the next state
		# Separate target network is used here to make training more stable
		next_q_values = torch.zeros(1)
		next_q_values = self.target_network(current_state * ~done_mask).max()

		# Calculate expected Q-values with immediate reward and
		# discounted expected return
		expected_q_values = reward + self.gamma * next_q_values

		loss = F.smooth_l1_loss(q_values, expected_q_values)

		self.optimizer.zero_grad()
		loss.backward()
		nn.utils.clip_grad_norm_(self.policy_network.parameters(), 1.0)
		self.optimizer.step()

		return loss.item()

	def update_target_network(self):
		self.target_network.load_state_dict(self.policy_network.state_dict())

	def sample_database(self):
		from .models import SnakeState

		pks = SnakeState.objects.values_list('pk', flat=True)
		random_object = SnakeState.objects.get(pk=random.choice(pks))
		
		current_state = np.zeros(self.input_dim)
		current_state[random_object.apples] = 0.2
		current_state[random_object.scissors] = 0.4
		current_state[random_object.walls] = 0.6
		current_state[random_object.snakehead] = 0.8
		current_state[random_object.snakebody] = 1.0

		action = random_object.action

		next_state = np.zeros(self.input_dim)
		next_state_object = SnakeState.objects.filter(
			game_id=random_object.game_id,
			order_id=random_object.order_id + 1)

		reward = 0
		done = True

		if (next_state_object):
			next_state[next_state_object[0].apples] = 0.2
			next_state[next_state_object[0].scissors] = 0.4
			next_state[next_state_object[0].walls] = 0.6
			next_state[next_state_object[0].snakehead] = 0.8
			next_state[next_state_object[0].snakebody] = 1.0
			reward = next_state_object[0].score - random_object.score
			done = False

		if (random_object.snake_dead):
			reward -= 5
		elif (random_object.snake_fed):
			reward += 20

		return current_state, action, next_state, reward, done