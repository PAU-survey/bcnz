#!/use/bin/env python
# encoding: UTF8

def assert_compat(conf):
    # Options not supported in BCNZ
    assert not conf['color']
    assert not conf['z_thr']
    assert conf['photo_errors']
    assert conf['n_peaks'] == 1
    assert not conf['convolve_p']
    assert not conf['probs']
    assert not conf['probs2']
    assert conf['probs_lite']
    assert not conf['nmax']
    assert not conf['zc']
    assert not conf['add_spec_prob']
    assert not conf['plots']
    assert not conf['interactive']
    assert not conf['merge_peaks']
