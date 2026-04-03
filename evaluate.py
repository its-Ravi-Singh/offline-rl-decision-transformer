import numpy as np
import torch
from collections import Counter

from utils.visualize import record_episode


def evaluate(model, env, num_episodes=20):

    model.eval()

    episode_returns = []
    all_actions = []

    for ep in range(num_episodes):

        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:

            action = model.act(state)

            all_actions.append(action)

            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            total_reward += reward

        episode_returns.append(total_reward)

    # ---------- Metrics ----------
    avg_return = np.mean(episode_returns)
    std_return = np.std(episode_returns)

    print("\n===== EVALUATION REPORT =====")
    print(f"Episodes: {num_episodes}")
    print(f"Average Return: {avg_return:.2f}")
    print(f"Std Return: {std_return:.2f}")

    # ---------- Action Distribution ----------
    action_counts = Counter(all_actions)
    total_actions = sum(action_counts.values())

    print("\nAction Distribution:")
    for action, count in action_counts.items():
        print(f"Action {action}: {count} ({count/total_actions:.2%})")

    # ---------- Policy Behavior Diagnosis ----------
    if len(action_counts) == 1:
        print("\n⚠️ Model is collapsing to a single action → no learning")
    elif max(action_counts.values()) / total_actions > 0.9:
        print("\n⚠️ Highly biased policy → still near-random or collapsed")
    else:
        print("\n✅ Policy has some diversity")

    # ---------- Performance Diagnosis ----------
    if avg_return < 20:
        print("\n❌ Model is behaving like a random policy")
        print("Possible issues:")
        print("- Training data is random (no expert supervision)")
        print("- Model underfitting")
        print("- Labels not informative")

    elif avg_return < 100:
        print("\n⚠️ Model learned something but still weak")

    else:
        print("\n✅ Strong policy")

    # ---------- Save rollout video ----------
    record_episode(env, model)

    return {
        "avg_return": avg_return,
        "std_return": std_return,
        "action_distribution": dict(action_counts),
    }