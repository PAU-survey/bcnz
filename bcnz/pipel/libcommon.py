#!/usr/bin/env python
# encoding: UTF8

# Unlike what the name say, this is not a dumping ground for general
# code. I am experimenting with not having to call a function for
# setting up a sub-pipeline in multiple locations.

import ipdb

class Common:
    # Test on how to assign pipelines....
    def __init__(self, name):
        self.name = name

def another(pipel, root=()):
    # Note: This is a debug method to be deleted...

    if isinstance(pipel, Common):
        print('.'.join(root)) 
        return

    for key,dep in pipel.depend.items(): 
        another(dep, root+(key,))

def replace(pipel, D, root=()):
    """Replace the common objects with the actual tasks."""

    for key,dep in pipel.depend.items(): 
        new_root = root+(key,)   
        if new_root == 'pzcat.galcat.input.input.cfht_cat.parent_cat':
            ipdb.set_trace()

        if isinstance(dep, Common):
            if not dep.name in D:
                ipdb.set_trace()
                raise KeyError('.'.join(new_root))

            new_job = D[dep.name]
            pipel.depend[key] = new_job

            replace(new_job, D, new_root)
        else:
            replace(dep, D, new_root)
