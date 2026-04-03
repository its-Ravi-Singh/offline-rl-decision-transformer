import cv2
import numpy as np

def record_episode(env, model, path="results/rollout.mp4"):

    frames = []

    state, _ = env.reset()

    for _ in range(200):

        frame = env.render()

        # Ensure frame is a numpy array
        frame = np.array(frame)

        frames.append(frame)

        action = model.act(state)

        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        if done:
            break

    # Ensure frames exist
    if len(frames) == 0:
        print("No frames captured!")
        return

    # Get frame shape safely
    h, w = frames[0].shape[:2]

    video = cv2.VideoWriter(
        path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        30,
        (w, h)
    )

    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video.write(frame_bgr)

    video.release()
    print(f"Saved video to {path}")