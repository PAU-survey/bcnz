# Input data
from .all_filters import all_filters
from .extinction_laigle import extinction_laigle
from .load_seds import load_seds
from .line_ratios import line_ratios

# Core model estimation.
from .model_cont import model_cont
from .model_lines import model_lines

# Adjust and rebin.
from .fmod_adjust import fmod_adjust
from .rebin import rebin

def model_single(seds, ext_law, EBV, sep_OIII, sed_dir, use_lines):
    """Create a single model."""

    seds_cont = load_seds(sed_dir)

    ratios = line_ratios()
    filters = all_filters()
    extinction = extinction_laigle()

    # Continuum and lines.
    model_cont_df = modelcont(filters, seds_cont, extinction, sed_vals=seds, EBV=EBV, ext_law=ext_law)
    model_lines_df = model_lines(ratios, filters, extinction, EBV=EBV, ext_law=ext_law, sed_vals=seds)

    model_orig = fmod_adjust(model_cont_df, model_lines_df)
    model_binned = rebin(model_orig)

    return model_binned
