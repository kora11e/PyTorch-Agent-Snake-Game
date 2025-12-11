import torch
import random
import pygame
import numpy as np
from collections import deque
from snake_game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 1e-4
GAMMA = 0.99
TARGET_UPDATE = 500 

class Agent:
    def __init__(self):
        self.n_games = 0
        self.gamma = GAMMA
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.memory = deque(maxlen=MAX_MEMORY)

        self.model = Linear_QNet(11, 256, 3)
        self.target_model = Linear_QNet(11, 256, 3)
        self.update_target_model()

        self.trainer = QTrainer(self.model, self.target_model, lr=LR, gamma=self.gamma)
        self.step_count = 0

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def getState(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y + 20)
        point_d = Point(head.x, head.y - 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),

            dir_l,
            dir_r,
            dir_u,
            dir_d,

            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y   # food down
        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, np.argmax(action), reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        loss = self.trainer.train_step(batch)
        return loss

    def train_short_memory(self, state, action, reward, next_state, done):
        batch = [(state, np.argmax(action), reward, next_state, done)]
        self.trainer.train_step(batch)

    def get_action(self, state):
        final_move = [0, 0, 0]

        if random.random() < self.epsilon:
            move = random.randint(0, 2)
        else:
            state0 = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()

        final_move[move] = 1
        return final_move

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def train():
    plot_scores, plot_mean_scores = [], []
    total_score, record = 0, 0
    agent = Agent()
    game = SnakeGameAI()

    while True:
        state_old = agent.getState(game)
        final_move = agent.get_action(state_old)

        reward, done, score = game.play_step(final_move)
        pygame.display.update()  # force refresh
        pygame.time.delay(20) 
        state_new = agent.getState(game)

        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            final_score = score

            agent.n_games += 1
            loss = agent.train_long_memory()
            agent.decay_epsilon()

            if agent.n_games % TARGET_UPDATE == 0:
                agent.update_target_model()

            if final_score > record:
                record = final_score
                agent.model.save()

            print(f"Game {agent.n_games} | Score {final_score} | Record {record} | Eps {agent.epsilon:.3f}")

            plot_scores.append(final_score)
            total_score += final_score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

            game.reset()

if __name__ == '__main__':
    train()
