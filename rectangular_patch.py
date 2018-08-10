# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 14:26:16 2018

@author: eenvdo
"""

import patch_antenna_calculator as pac


# for bandwidth calculations
VSWR = 2

# center frequency
freq = 16.0  # centre frequency in GHz
freq *= 1e9  # convert to Hz
lambda_0 = pac.c0 / freq  # corresponding wavelength in m

# substrate parameters
subst_h = 50.0  # substrate thickness in um
subst_h *= 1e-6  # convert to base units
subst_er = 3.5  # relative permittivity
subst_ur = 1  # relative permeability
subst_tand = 0.008  # loss tangent

# conductor parameters
metal_cond = 4.1e7  # metal conductivity in S/m
metal_ur = 0.99996  # metal relative permeability

rect_patch = pac.patch_design_initial(freq, subst_er, subst_h)

rect_patch_zin_q_aeh = pac.calc_zin_q_aeh(freq, lambda_0, subst_h, subst_er,
                                          subst_ur, subst_tand, metal_cond,
                                          metal_ur,
                                          rect_patch['patch_l'].value,
                                          rect_patch['patch_w'].value,
                                          rect_patch['delta_patch_l'].value)

rect_patch_zin_q_atma = pac.calc_zin_q_atma(freq, lambda_0, subst_h, subst_er,
                                            subst_tand, metal_cond, metal_ur,
                                            rect_patch['patch_l'].value,
                                            rect_patch['patch_w'].value,
                                            rect_patch['delta_patch_l'].value)

rect_patch_zin_q_mpad = pac.calc_zin_q_mpad(freq, lambda_0, subst_h, subst_er,
                                            subst_ur, subst_tand, metal_cond,
                                            metal_ur,
                                            rect_patch['patch_l'].value,
                                            rect_patch['patch_w'].value,
                                            rect_patch['delta_patch_l'].value)

rect_patch_zin_q_atad = pac.calc_zin_q_atad(freq, lambda_0, subst_h, subst_er,
                                            subst_tand, metal_cond, metal_ur,
                                            rect_patch['patch_l'].value,
                                            rect_patch['patch_w'].value,
                                            rect_patch['delta_patch_l'].value)
