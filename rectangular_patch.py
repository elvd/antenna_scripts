# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 14:26:16 2018

@author: eenvdo
"""

import patch_antenna_calculator as pac


VSWR = 2

# center frequency
freq = 5.15  # centre frequency in GHz
freq *= 1e9  # convert to Hz
lambda_0 = pac.c0 / freq  # corresponding wavelength in m

# substrate parameters
subst_h = 780  # substrate thickness in um
subst_h *= 1e-6  # convert to base units
subst_er = 2.98  # relative permittivity
subst_ur = 1  # relative permeability
subst_tand = 0.0027  # loss tangent

# conductor parameters
metal_cond = 5.88e7  # metal conductivity in S/m
metal_ur = 1.0  # metal relative permeability
