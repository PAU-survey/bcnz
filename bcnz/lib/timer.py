#!/usr/bin/env python
# encoding: UTF8

import time

class watch:
    """Stopwatch that can disply time intervals."""

    def __init__(self, fmt='%H:%M:%S'):
        self.fmt = fmt
        self.start_time = time.time()

    def __repr__(self):
        """Output time used."""

        diff = time.gmtime(time.time() - self.start_time)

        days = diff[2]
        if 1 < days:
            return 'The runtime is in addition {0} days' % str(days-1)

        time_str = time.strftime(self.fmt, diff)
 
        return '\nElapsed time %s\n' % time_str
