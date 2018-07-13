# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 12:54:08 2018

@author: eenvdo
"""

import patch_antenna_calculator as pac


VSWR = 2

# center frequency
freq = 30.0  # centre frequency in GHz
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


circ_patch = pac.circ_patch_design(freq, subst_er, subst_h)
circ_patch_radius = circ_patch['radius'].value

circ_patch_zin_q = pac.calc_zin_q_circ_patch(freq, lambda_0, subst_h, subst_er,
                                             subst_ur, subst_tand, metal_cond,
                                             metal_ur, circ_patch_radius)

circ_patch_q = circ_patch_zin_q['q'].value

circ_patch_bw = pac.calc_patch_bandwidths(VSWR, circ_patch_q, subst_er,
                                          subst_h, lambda_0)

print('Results for circular patch')
print('-'*20)
print('Design parameters')
print('-'*20)
print('Centre frequency, GHz: {:.3f}'.format(freq))
print('Substrate thickness, um: {:.3f}'.format(subst_h))
print('Substrate Dk: {:.3f}'.format(subst_er))
print('-'*20)
print('Results')
print('-'*20)
print('Radius, mm: {:.3f}'.format(circ_patch_radius*1e3))
print('Overall Q factor: {:.3f}'.format(circ_patch_q))
print('Inset distance for 50 Ohm, mm: {:.3f}'.
      format((circ_patch_radius-circ_patch_zin_q['r_optim'].value)*1e3))
print('Bandwidth for linear polarisaion: {:.3f}'.
      format(circ_patch_bw['BW_linear'].value*100))
