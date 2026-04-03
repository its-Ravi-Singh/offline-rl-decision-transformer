import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from models.decision_transformer import PolicyNet
from data.dataset import TrajectoryDataset
from train import train
from evaluate import evaluate
from utils.check import save_model

def generate_data(env, num_episodes=100):

    data = []

    for _ in range(num_episodes):

        states = []
        actions = []

        state, _ = env.reset()

        for _ in range(200):

            action = env.action_space.sample()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            states.append(state)
            actions.append(action)

            state = next_state

            if done:
                break

        data.append((np.array(states), np.array(actions)))

    return data


def main():

    env = gym.make("CartPole-v1", render_mode="rgb_array")

    print("Collecting dataset...")
    data = generate_data(env)

    dataset = TrajectoryDataset(data)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = PolicyNet(state_dim=4, act_dim=2)

    print("Training...")
    train(model, dataloader, epochs=5)

    print("Evaluating...")
    evaluate(model, env)

    save_model(model, "saved_models/decision_transformer.pth")


if __name__ == "__main__":
    main()