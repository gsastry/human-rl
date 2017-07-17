import numpy as np  # isort:skip
import multiprocessing

ctx = multiprocessing.get_context("forkserver")

def input_thread_fn(queue, dataloader, batch_size, episode_paths):
    eps = np.random.choice(episode_paths, size=1)
    next_X, next_y = dataloader.load_features_and_labels(eps)
    # batch_size = min(batch_size, len(episode_paths))
    while(True):
        if len(next_y) >= batch_size * 2:
            p = np.random.permutation(len(next_y))
            for k in next_X.keys():
                next_X[k] = next_X[k][p]
            next_y = next_y[p]
        while len(next_y) >= batch_size * 2:
            X2 = {k: v[:batch_size] for k, v in next_X.items()}
            y2 = next_y[:batch_size]
            queue.put((X2, y2))
            for k in next_X.keys():
                next_X[k] = next_X[k][batch_size:]
            next_y = next_y[batch_size:]
        eps = np.random.choice(episode_paths, size=1)
        X, y = dataloader.load_features_and_labels(eps)
        for k in next_X.keys():
            next_X[k] = np.concatenate([next_X[k], X[k]], axis=0)
        next_y = np.concatenate([next_y, y], axis=0)
