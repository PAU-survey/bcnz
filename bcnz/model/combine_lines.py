# Copyright (C) 2020 Martin B. Eriksen
# This file is part of BCNz <https://github.com/PAU-survey/bcnz>.
#
# BCNz is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# BCNz is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with BCNz.  If not, see <http://www.gnu.org/licenses/>.
#!/usr/bin/env python
# encoding: UTF8

# Also something from Alex?

def combine_lines(model_lines, line_ratios):
    """Combine several individual line fluxes together given
     input ratios

       Args:
           model_lines (df): Emission line model.
           line_ratios (dict): Ratio coefficients for each line
    """

    i = 0
    for line_key, ratio in line_ratios.items():
        # test.loc[test.sed == line_key, "flux"] *= ratio
        sub = model_lines.loc[model_lines.sed == line_key]
        sub.loc[:, "flux"] *= ratio
        sub = sub.drop(columns="sed")
        sub = sub.set_index(["z", "band", "ext_law", "EBV"])
        if i == 0:
            output = sub.copy()
        else:
            output = output.add(sub)
        i = i + 1

    output["sed"] = "lines"
    output = output.reset_index()

    return output
