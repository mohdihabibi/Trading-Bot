import random
from tensorboardX import SummaryWriter
from collections import deque

import numpy as np
from utils.model import model, load
from keras.models import clone_model


class Agent:
    """ Stock Trading Bot """

    def __init__(self, current_price, pretrained=False):
        # agent config
        self.state_size = 5    	# normalized previous days
        self.cash_in_hand = 6000
        self.total_share = 20
        self.action_size = 3           		# [sit, buy, sell]
        self.inventory = []
        self.memory = deque(maxlen=10000)
        self.first_iter = True
        self.initial_price = current_price
        self.writer = SummaryWriter()
        if pretrained:
            self.model = load()
        else:
            self.model = model()
        for i in range(20):
            self.inventory.append(self.initial_price)
        self.gamma = 0.95  # affinity for long term reward
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.n_iter = 1
        self.reset_every = 1000
        self.target_model = clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())

    def reset(self):
        self.cash_in_hand = 6000
        self.total_share = 20
        self.inventory = []
        for i in range(20):
            self.inventory.append(self.initial_price)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, is_eval=False):
        # take random action in order to diversify experience at the beginning
        if not is_eval and random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        if self.first_iter:
            self.first_iter = False
            return 1  # make a definite buy on the first iter

        action_probs = self.model.predict(state)
        return np.argmax(action_probs[0])

    def train_experience_replay(self, batch_size):
        """Train on previous experiences in memory
        """
        mini_batch = random.sample(self.memory, batch_size)
        X_train, y_train = [], []
        self.n_iter += 1
        if self.n_iter % self.reset_every == 0:
            # reset target model weights
            self.target_model.set_weights(self.model.get_weights())

        for state, action, reward, next_state, done in mini_batch:
            if done:
                target = reward
            else:
                # approximate double deep q-learning equation
                target = reward + self.gamma * \
                    self.target_model.predict(next_state)[0][np.argmax(
                        self.model.predict(next_state)[0])]

            # estimate q-values based on current state
            q_values = self.model.predict(state)
            # update the target for current action based on discounted reward
            q_values[0][action] = target

            X_train.append(state[0])
            y_train.append(q_values[0])

        # update q-function parameters based on huber loss gradient
        loss = self.model.fit(
            np.array(X_train), np.array(y_train),
            epochs=1, verbose=0
        ).history["loss"][0]
        self.writer.add_scalar('loss per iteration', loss, self.n_iter)
        # as the training goes on we want the agent to
        # make less random and more optimal decisions
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss
