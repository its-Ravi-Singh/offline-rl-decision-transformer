import numpy as np
import gymnasium as gym


def evaluate(
    model,
    env_name="Hopper-v4",
    num_episodes=20,
    target_rtg_max=3000.0,
    record_video=True,
):

    env = gym.make(env_name, render_mode="rgb_array")
    video_enabled = False
    if record_video:
        try:
            env = gym.wrappers.RecordVideo(
                env,
                video_folder="results",
                episode_trigger=lambda x: x == 0,
                disable_logger=True,
            )
            video_enabled = True
        except gym.error.DependencyNotInstalled as exc:
            print(f"Video recording disabled: {exc}")

    returns = []

    model.eval()


    for _ in range(num_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0


        rtg = 1.0
        state_history = [obs.astype(np.float32)]
        action_history = [np.zeros(getattr(model, "act_dim", env.action_space.shape[0]), dtype=np.float32)]
        rtg_history = [rtg]
        done = False

        while not done:
            try:
                a = model.act(
                    obs.astype(np.float32),
                    target_rtg=rtg,
                    state_history=state_history,
                    action_history=action_history,
                    rtg_history=rtg_history,
                )
            except TypeError:
                a = model.act(obs.astype(np.float32), target_rtg=rtg)


            a = np.clip(a, -1.0, 1.0)

            obs, rew, term, trunc, _ = env.step(a)
            ep_reward += rew


            rtg -= rew / (target_rtg_max + 1e-8)
            state_history.append(obs.astype(np.float32))
            action_history.append(a.astype(np.float32))
            rtg_history.append(float(rtg))
            done = term or trunc

        returns.append(ep_reward)

    env.close()

    mean_r = np.mean(returns)
    std_r  = np.std(returns)

    print(f"\nAverage Score: {mean_r:.1f} | Std Dev: {std_r:.1f}")
    print(f"Min/Max Returns: {min(returns):.0f} / {max(returns):.0f}")
    if video_enabled:
        print(f"Saved video to results/ folder!")

    return {
        "avg_return": float(mean_r),
        "std_return": float(std_r),
        "min_return": float(np.min(returns)),
        "max_return": float(np.max(returns)),
        "returns": [float(value) for value in returns],
    }
