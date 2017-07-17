#!/usr/bin/env python3

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "humanrl"))

if __name__ == '__main__':
    import multiprocessing

    from humanrl import frame
    import tqdm

    episode_paths = frame.episode_paths("logs/")
    print(len(episode_paths))
    # frame.fix_episode(episode_paths[0])
    pool = multiprocessing.Pool(multiprocessing.cpu_count() - 1)
    # for _ in tqdm.tqdm(
    #         pool.imap_unordered(frame.check_episode, episode_paths), total=len(episode_paths)):
    #     pass
# %timeit frame.fix_episode(episode_paths[0])
