# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import time
import logging
from rich.console import Console
from rich.logging import RichHandler

def get_time(sec):
    h = int(sec//3600)
    m = int((sec//60)%60)
    s = int(sec%60)
    return h,m,s

class TimeFilter(logging.Filter):

    def filter(self, record):
        try:
          start = self.start
        except AttributeError:
          start = self.start = time.time()

        time_elapsed = get_time(time.time() - start)

        record.relative = "{0}:{1:02d}:{2:02d}".format(*time_elapsed)

        # self.last = record.relativeCreated/1000.0
        return True

class Logger(object):
    def __init__(self, rank=0, log_dir=".log"):
        # other libraries may set logging before arriving at this line.
        # by reloading logging, we can get rid of previous configs set by other libraries.
        from importlib import reload
        reload(logging)
        self.rank = rank
        if self.rank == 0:
            os.makedirs(log_dir, exist_ok=True)

            log_file = open(os.path.join(log_dir, "log.txt"), "w")
            file_console = Console(file=log_file, width=150)
            logging.basicConfig(
                level=logging.INFO,
                format="(%(relative)s) %(message)s",
                datefmt="[%X]",
                force=True,
                handlers=[
                    RichHandler(show_path=False),
                    RichHandler(console=file_console, show_path=False)
                ],
            )
            # https://stackoverflow.com/questions/31521859/python-logging-module-time-since-last-log
            log = logging.getLogger()
            [hndl.addFilter(TimeFilter()) for hndl in log.handlers]

    def info(self, string, *args):
        if self.rank == 0:
            logging.info(string, *args)

    def warning(self, string, *args):
        if self.rank == 0:
            logging.warning(string, *args)

    def error(self, string, *args):
        if self.rank == 0:
            logging.error(string, *args)
