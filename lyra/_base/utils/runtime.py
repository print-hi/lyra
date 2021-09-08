import sys
import time

import contextlib
from tqdm import tqdm

class writeToStdOut(object):
    def __init__(self, file):
        self.file = file

    def write(self, message):
        if len(message.rstrip()) > 0:
            tqdm.write(message, file=self.file)

@contextlib.contextmanager
def writeout():
    out = sys.stdout
    sys.stdout = writeToStdOut(out)
    yield
    sys.stdout = out

def timer(f):
    def run(*args, **kwargs):
        before = time.time()
        f()
        print("Elapsed Time:", time.time() - before)
    return run
