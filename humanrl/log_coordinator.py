import glob
import os
import pathlib
import re
import shutil

import numpy as np


class LogCoordinator:
    def __init__(self,
                 max_length,
                 logdir,
                 filename_str,
                 resume_if_logdir_exists=True,
                 initial_interval=1):
        """
        To record every episode, set max_length to None. To resume, rather than overwriting files,
        set resume_if_logdir_exists to True
        """
        self.max_length = max_length
        self.interval = initial_interval
        self.i = 0
        self.logdir = logdir
        self.filename_str = filename_str

        if os.path.exists(self.logdir):
            if resume_if_logdir_exists:
                i, interval = self.existing_index_and_interval()
                if i is not None:
                    self.i, self.interval = i + 1, interval
            else:
                shutil.rmtree(self.logdir)
                os.makedirs(self.logdir)
        else:
            os.makedirs(self.logdir)

    def get_filename(self):
        return self._get_filename(self.i)

    def _get_filename(self, i):
        return os.path.join(self.logdir, self.filename_str.format(i=i))

    def clean(self):
        if self.max_length is not None:
            for i, f in self.existing_indices_and_files():
                if i % (self.interval * 2) != 0:
                    os.remove(f)

    def should_log(self):
        return (self.i % self.interval == 0)

    def step(self):
        if self.max_length is not None:
            if self.i == self.interval * self.max_length and self.i > 0:
                self.clean()
                self.interval *= 2
        self.i += 1

    def existing_indices_and_files(self):
        path_str = os.path.join(self.logdir, self.filename_str)
        existing_files = glob.glob(os.path.join(path_str.replace("{i}", "*")))
        L = []
        pattern = re.compile(path_str.replace("{i}", "(.*)"))
        for f in existing_files:
            m = pattern.match(f)
            if m:
                i = int(m.group(1))
                L.append((i, f))
        return sorted(L)

    def existing_index_and_interval(self):
        indices = [i for i, f in self.existing_indices_and_files()]
        if len(indices) == 0:
            return None, 1
        elif len(indices) == 1:
            return indices[0], 1
        indices.sort()
        diff = np.diff(indices)
        interval = diff[0]
        return max(indices), interval


if __name__ == "__main__":
    lc = LogCoordinator(None, "/tmp/foolog", "yo{i}.txt")

    for i in range(100):
        if lc.should_log():
            print(lc.get_filename())
            pathlib.Path(lc.get_filename()).touch()
        lc.step()
    print(lc.existing_index_and_interval())
    print(lc.existing_indices_and_files())
