import os
import sys
import time
from datetime import datetime

class TextLogger(object):
    """Writes stream output to external text file.

    Args:
        filename (str): the file to write stream output
        stream: the stream to read from. Default: sys.stdout
    """
    def __init__(self, filename, stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.terminal.close()
        self.log.close()


class CompleteLogger:
    """
    A useful logger that

    - writes outputs to files and displays them on the console at the same time.
    - manages the directory of checkpoints and debugging images.

    Args:
        root (str): the root directory of logger
        phase (str): the phase of training.

    """

    def __init__(self, root, phase='train'):
        self.root = root
        self.phase = phase





        timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        log_filename = os.path.join(self.root, "log_{}-{}.txt".format(phase, timestamp))


        self.logger = TextLogger(log_filename)
        sys.stdout = self.logger
        sys.stderr = self.logger

    def close(self):
        sys.stdout = sys.stdout
        sys.stderr = sys.stderr
        self.logger.close()
