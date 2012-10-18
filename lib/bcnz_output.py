#!/use/bin/env python
# encoding: UTF8
import os
import pdb
import shutil
import time

def output_file(output_file):
    """Create file object to write BPZ catalog."""

    file_name = output_file
    if os.path.exists(file_name):
        dst = '%s.bak' % file_name
        shutil.move(file_name, dst)

        print('File %s exists. Moving it to %s.' % (file_name, dst))

    if os.path.exists(file_name):
        raise SystemError

    return open(file_name, 'w')

def create_header(conf, obs_file):
    """Header with the parameters and catalog fields."""
 
    fmt = '%Y:%m:%d %H:%M'
    time_str = time.strftime(fmt, time.localtime())

    # Input file and current time
    header = ['# {0} {1}'.format(obs_file, time_str), '#']

    conf_keys = conf.keys()
    conf_keys.sort()
    max_key_len = max(len(x) for x in conf_keys)

    for key in conf_keys:
        header.append('# {0} =  {1}'.format(key.ljust(max_key_len), \
                                           conf[key]))
    header.append('#')
    header.append('# Column information')
    for i, col in enumerate(conf['order']+conf['others']):
        header.append('# {0} {1}'.format(i+1, col))

    header.append('#')

    return header
