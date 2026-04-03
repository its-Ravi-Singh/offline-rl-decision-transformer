def run_episode(env, model, max_steps=200):

    state, _ = env.reset()
    total_reward = 0

    for _ in range(max_steps):

        action = model.act(state)

        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        total_reward += reward

        if done:
            break

    return total_reward