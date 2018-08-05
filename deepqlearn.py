import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras import backend as K
from keras.layers.core import Flatten

import os
EPISODES = 100 # Global Variable


class QLearningAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size 
        self.memory = deque(maxlen=100000) # deque is a sort of list with O(1) complexity
        self.gamma = 0.95    # discount rate for bellman equations, making it episodic
        self.epsilon = 1.0  # exploration rate greedy policy
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99 # rate at which decay follows 
        self.learning_rate = 0.01 # minimum value
        self.model = self.build_model() #Neural network is build in this panel
        self.target_model = self.build_model()
        self.update_target_model()

    def losscalc(self, target, prediction):
        # we can't use tensorflow huber loss as we have no error calculated
        error = prediction - target
        return K.mean(K.sqrt(1+K.square(error))-1, axis=-1)

    def build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Conv2D(128, kernel_size=(6,6), activation = 'relu')
        model.add(Conv2Ds(256, kernel_size=(6,6) activation='relu'))
        model.add(MaxPooling2D(strides=(6,6)))
        model.add(Flatten)
        model.add(Dense(1000, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax'))
        model.compile(loss=self.losscalc,
                      optimizer=Adam(lr=self.learning_rate))
        
        return model

        

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        #Here the agent remembers the last state of the game and based on that takes the new way
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Here the agent acts, this is the logic
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size) #range 
        act_values = self.model.predict(state) #here from the model that we have made, we are predicting the values
        return np.argmax(act_values[0])  # returns action (the max value)

    def replay(self, batch_size):
        #making the replay based on mini batch sizes
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                a = self.model.predict(next_state)[0]
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * t[np.argmax(a)]
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    

if __name__ == "__main__":
    env = gym.make('MsPacman-ram-v0')
    state_size = env.observation_space.shape[0] #returns the total possible states
    action_size = env.action_space.n #returns what are the possible actions
    print(action_size)
    print(state_size)
    agent = QLearningAgent(state_size, action_size) # Q learning class
    done = False
    batch_size = 128
    num=0
    f = open('output1.txt','w')
    for e in range(EPISODES):
        state = env.reset() # resets the environment everytime the new episode starts.
        state = np.reshape(state, [1, state_size])
        num2=0
        num=num+1
        for time in range(500):
            env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            print("episode:{}".format(num))
            print(reward)
            print("--ver:{}".format(num2))
            num2=num2+1
            next_state = np.reshape(next_state, [1, state_size]) # next state takes the shape 
            agent.remember(state, action, reward, next_state, done) # agent remembers the previous state, and based on that, moves through
            state = next_state
            if done: #if it goes to the next step
                
                f.write("episode: {}/{}, score: {}, e: {:.2}\n"
                      .format(e, EPISODES, time, agent.epsilon))

                break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
    f.close()
      