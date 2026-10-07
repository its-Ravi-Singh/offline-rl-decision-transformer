import numpy as np
import gymnasium as gym


# runs the model in the gym environment and returns episode stats
# record_video=True saves a video to results/ folder
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
                episode_trigger=lambda x: x == 0,  # only record first episode
                disable_logger=True,
            )
            video_enabled = True
        except gym.error.DependencyNotInstalled as exc:
            print(f"cant record video: {exc}")

    returns = []
    model.eval()

    for ep in range(num_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0

        # normalize rtg same way we did during training
        rtg = 1.0
        state_history = [obs.astype(np.float32)]
        action_history = []
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
                # perception transformer doesnt take history
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
        # print(f"  ep {ep+1}: {ep_reward:.1f}")  # uncomment to see per-ep scores

    env.close()

    mean_r = np.mean(returns)
    std_r = np.std(returns)
    print(f"\navg return: {mean_r:.1f} (+/- {std_r:.1f})")
    print(f"min/max: {min(returns):.0f} / {max(returns):.0f}")
    if video_enabled:
        print("video saved to results/")

    return {
        "avg_return": float(mean_r),
        "std_return": float(std_r),
        "min_return": float(np.min(returns)),
        "max_return": float(np.max(returns)),
        "returns": [float(r) for r in returns],
    }
