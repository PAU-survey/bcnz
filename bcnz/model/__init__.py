from .all_filters import all_filters
from .extinction_laigle import extinction_laigle
from .seds import seds as get_seds
from .line_ratios import line_ratios
from .emission_lines import emission_lines
from .ab_cont import ab_cont
from .fmod_adjust import fmod_adjust
from .rebin import rebin

def model_single(seds, ext_law, EBV, sep_OIII, sed_dir, use_lines):
    """Create a single model."""

    seds_cont = get_seds(sed_dir)

    ratios = line_ratios()
    filters = all_filters()
    extinction = extinction_laigle()

    # Continuum and lines.
    model_cont = ab_cont(filters, seds_cont, extinction, sed_vals=seds, EBV=EBV, ext_law=ext_law)
    model_lines = emission_lines(ratios, filters, extinction, EBV=EBV, ext_law=ext_law, sed_vals=seds)

    model_orig = fmod_adjust(model_cont, model_lines)
    model_binned = rebin(model_orig)

    return model_binned
