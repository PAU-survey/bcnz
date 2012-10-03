#!/use/bin/env python
import os
import pdb
import shutil

def output_file(output_file):
    """Create file object to write BPZ catalog."""

    file_name = output_file
#    if os.path.exists(file_name):
#        dst = '%s.bak' % file_name
#        shutil.move(file_name, dst)
#
#        print('File %s exists. Moving it to %s.' % (file_name, dst))

    if os.path.exists(file_name):
        raise SystemError

    return open(file_name, 'w')

