"""
Created on Sun Mar 19 18:05:58 2017

Design of standard gain pyramidal horn antennas. Uses two different approaches,
one from Antenna Engineering Handbook and one from Antenna Theory Analysis and
Design. Results obtained are slightly different for the same gain. After 3D EM
simulations in HFSS, the main difference is in the Side Lobe Level.

@author: viktor
"""
import numpy as np
import scipy.constants
from scipy.optimize import newton

# feed waveguide dimensions in metres and geometric mean frequencies in Hz
wvg_info = {'wr28': {'a': 7.112e-3, 'b': 3.556e-3, 'freq': 32.56e9},
            'wr10': {'a': 2.54e-3, 'b': 1.27e-3, 'freq': 90.83e9},
            'wr01': {'a': 254e-6, 'b': 127e-6, 'freq': 908.3e9},
            'wr90': {'a': 2.286e-2, 'b': 1.016e-2, 'freq': 10.08e9}}

# pick one to design for
wvg = 'wr01'

gain_dB = 30.0  # required boresight gain in dB
gain = np.power(10, gain_dB/10)  # convert to absolute numbers
wavelength = scipy.constants.c / wvg_info[wvg]['freq']  # in metres

# Parameters computed using algorithm in Antenna Engneering Handbook

# maximum normalised cylindrical-wave errors
delta_e_max = 0.2
delta_h_max = 0.2

g_1 = 2 * np.pi
alpha_1 = 3 * wavelength
beta_1 = 2 * wavelength

c_0 = (gain * np.power(wavelength, 2)) * alpha_1 / (g_1 * beta_1)

P_1 = c_0 / 12.0
P_2 = gain * np.power(wavelength, 2) / g_1
P_3 = wvg_info[wvg]['a'] * wvg_info[wvg]['b'] / 4.0
P = P_1 * (P_2 - P_3)

Q_1 = np.power(c_0, 2) / 128.0
Q_2 = np.power(wvg_info[wvg]['a'], 2) * beta_1 / alpha_1
Q = Q_1 * (Q_2 - np.power(wvg_info[wvg]['b'], 2))

u_1 = np.sqrt(np.power(Q, 2) + np.power(P, 3))
u_2 = np.abs(Q + u_1)
u = np.power(u_2, 1.0/3)

A2 = (u - P/u) + (np.power(wvg_info[wvg]['a'], 2) / 8.0)

A1_1 = 3 * np.power((u + P/u), 2)
A1 = np.sqrt(np.power(A2, 2) + A1_1)

a_1 = np.sqrt(A1 + A2)
a_2 = wvg_info[wvg]['a'] / 4.0
a_3 = wvg_info[wvg]['b'] * c_0 - np.power(wvg_info[wvg]['a'], 2) / 8.0
a = a_1 + a_2 - (a_3 / (4 * A1))  # aperture dimension a, in metres

b = gain * np.power(wavelength, 2) / (g_1 * a)  # aperture dimension b, in m

Lh = np.power(a, 2) / alpha_1
Le = np.power(b, 2) / beta_1

h_Lh = Lh * (1 - wvg_info[wvg]['a'] / a)  # flare length 1, in metres
h_Le = Le * (1 - wvg_info[wvg]['b'] / b)  # flare length 2, in metres

# calculate axial lengths corresponding to specified max phase error
Lh_max_error = np.power(a, 2) / (8 * delta_h_max * wavelength)
Le_max_error = np.power(b, 2) / (8 * delta_e_max * wavelength)

h_Lh_max_error = Lh_max_error * (1 - wvg_info[wvg]['a'] / a)
h_Le_max_error = Le_max_error * (1 - wvg_info[wvg]['b'] / b)

h_max_error = np.max([h_Lh_max_error, h_Le_max_error])

# phasing section length calculation
k = 2 * np.pi / wavelength
beta_10 = k * np.sqrt(1 - np.power(wavelength / (2 * b), 2))
beta_12 = k * np.sqrt(1 - np.power(wavelength / (2 * b), 2) *
                      (4 + np.power(b / a, 2)))

l_phase = (3 / 2 * np.pi) / (beta_10 - beta_12)

print('Dimensions using Antenna Engineering book')
print('-----------------------------------------')
print('Aperture dimensions: a = {:.3f} mm, b = {:.3f} mm'.format(a*1e3, b*1e3))
print('Axial length in H plane = {:.3f} mm'.format(h_Lh*1e3))
print('Axial length in E plane = {:.3f} mm'.format(h_Le*1e3))
print('Axial length for min error h = {:.3f} mm'.format(h_max_error*1e3))
print('Phasing section length l = {:.3f} mm'.format(l_phase*1e3))
print('-----------------------------------------')

# Parameters computed using algorithm in Antenna Theory


def right_side_func(chi):
    right_side_1 = np.power(gain, 2) / (6 * np.power(np.pi, 3) * chi) - 1
    right_side_2 = gain / (2 * np.pi)
    right_side_3 = np.sqrt(3 / (2 * np.pi))
    right_side_4 = 1.0 / np.sqrt(chi)
    right_side_5 = wvg_info[wvg]['a'] / wavelength
    right_side = np.power((right_side_2 * right_side_3 *
                           right_side_4 - right_side_5), 2) * right_side_1
    return right_side


def left_side_func(chi):
    left_side_1 = (2 * chi - 1)
    left_side_2 = np.sqrt(2 * chi) - wvg_info[wvg]['b'] / wavelength
    left_side = np.power(left_side_2, 2) * left_side_1
    return left_side


def func(chi):
    return right_side_func(chi) / left_side_func(chi) - 1


chi_initial = gain / (2 * np.pi * np.sqrt(2 * np.pi))
chi_optimised = newton(func, chi_initial)

rho_e = chi_optimised * wavelength
rho_h = wavelength * np.power(gain, 2) / \
        (chi_optimised * 8 * np.power(np.pi, 3))

# aperture dimensions
a1 = np.sqrt(3 * wavelength * rho_h)
b1 = np.sqrt(2 * wavelength * rho_e)

# axial lengths
p_e = (b1 - wvg_info[wvg]['b']) * np.sqrt(np.power(rho_e/b1, 2) - 0.25)
p_h = (a1 - wvg_info[wvg]['a']) * np.sqrt(np.power(rho_h/a1, 2) - 0.25)

# phasing section length calculation
k = 2 * np.pi / wavelength
beta_10 = k * np.sqrt(1 - np.power(wavelength / (2 * b1), 2))
beta_12 = k * np.sqrt(1 - np.power(wavelength / (2 * b1), 2) *
                      (4 + np.power(b1 / a1, 2)))

l_phase = (3 / 2 * np.pi) / (beta_10 - beta_12)

print('Dimensions using Antenna Theory book')
print('------------------------------------')
print('Aperture dimensions: a = {:.3f} mm, b = {:.3f} mm'.
      format(a1*1e3, b1*1e3))
print('Axial length h = {:.3f} mm'.format(p_e*1e3))
print('Phasing section length l = {:.3f} mm'.format(l_phase*1e3))
print('------------------------------------')
