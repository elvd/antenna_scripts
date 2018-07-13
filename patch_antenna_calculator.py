"""
Created on Fri Apr  6 13:38:01 2018

Design of rectangular patch antennas and investigation of their expected EM
performance as function of substrate height, metal thickness, and other
parameters. Incorporates formulas from several different books. Includes calc-
ulations of corner truncation needed for circular polarisation and for slots
for wideband operation.

@author: eenvdo
"""

from collections import namedtuple
import numpy as np
from scipy.special import jv, jnp_zeros
from scipy.integrate import quad
from scipy.optimize import newton
from scipy.constants import speed_of_light as c0
from scipy.constants import epsilon_0 as epsilon_0
from scipy.constants import mu_0 as mu_0


# characteristic impedance of vacuum
eta_0 = np.sqrt(mu_0 / epsilon_0)

# NamedTuple to store results
result = namedtuple('Result', 'description value')

# center frequency
freq = 5.2  # center frequency in GHz
freq *= 1e9  # convert to Hz
lambda_0 = c0 / freq  # corresponding wavelength in m

# substrate parameters
subst_h = 780  # substrate thickness in um
subst_h *= 1e-6  # convert to base units
subst_er = 2.98  # relative permittivity
subst_ur = 1  # relative permeability
subst_tand = 0.0027  # loss tangent

# conductor parameters
metal_cond = 5.88e7  # metal conductivity in S/m
metal_ur = 1.0  # metal relative permeability

# implantable antenna parameters
air_er = 1  # by definition
rat_skin_er = 15  # initial test value
rat_skin_h = 1.5  # in mm
rat_skin_h *= 1e-3


def circ_patch_design(freq, subst_er, subst_h):
    # assumes TM11 mode, i.e. fundamental mode
    n_index = 1
    m_index = 1

    A_nm = jnp_zeros(n_index, m_index)[0]

    patch_radius = A_nm * c0 / (2 * np.pi * freq * np.sqrt(subst_er))

    for _ in range(20):
        radius_new = np.pi * patch_radius / (2 * subst_h)
        radius_new = np.log(radius_new) + 1.7726
        radius_new *= (2 * subst_h / (np.pi * patch_radius * subst_er))
        radius_new += 1
        radius_new = np.power(radius_new, 0.5)
        radius_new *= (A_nm * c0 / (2 * np.pi * freq * np.sqrt(subst_er)))

        patch_radius = radius_new

    results_dict = {'radius': result('Circular patch radius', patch_radius)}

    return results_dict


def calc_i1(theta, n, beta_0, radius):
    bessel_func_arg = beta_0 * radius * np.sin(theta)

    i1_1 = jv(n+1, bessel_func_arg)
    i1_2 = jv(n-1, bessel_func_arg)

    i1_3 = np.power(i1_1 - i1_2, 2)

    i1_4 = np.power(np.cos(theta), 2) * i1_3

    i1 = (i1_3 + i1_4) * np.sin(theta)

    return i1


def calc_rad_cond_integral(theta, n, beta_0, radius):
    bessel_func_arg = beta_0 * radius * np.sin(theta)

    bp = jv(n-1, bessel_func_arg)
    bp += jv(n+1, bessel_func_arg)
    bm = jv(n-1, bessel_func_arg)
    bm -= jv(n+1, bessel_func_arg)

    integral = (np.power(bm, 2) + np.power(bp, 2) * np.power(np.cos(theta), 2))
    integral *= np.sin(theta)

    return integral


def calc_zin_q_circ_patch(freq, lambda_0, subst_h, subst_er, subst_ur,
                          subst_tand, metal_cond, metal_ur, radius):
    n = 1
    zin_goal = 50.0
    beta_0 = 2 * np.pi / lambda_0
    beta = 2 * np.pi / (lambda_0 / np.sqrt(subst_er))

    Q_dielectric = 1 / subst_tand

    Q_conductor = subst_h * np.sqrt(np.pi * freq * metal_cond * metal_ur *
                                    mu_0)

    i1 = quad(calc_i1, 0, np.pi, args=(n, beta_0, radius))[0]

    foo = 240 * (np.power(beta * radius, 2) - np.power(n, 2))
    bar = subst_h * mu_0 * metal_ur * freq * i1 * np.power(beta_0 * radius, 2)

    Q_radiation = foo / bar

    Q = 1 / Q_radiation + 1 / Q_dielectric + 1 / Q_conductor
    Q = 1 / Q

    loss_dielectric = subst_tand / (4 * mu_0 * subst_h * freq)
    loss_dielectric *= (np.power(beta * radius, 2) - np.power(n, 2))

    loss_conductor = np.pi * np.power(np.pi * freq * mu_0, -1.5)
    loss_conductor /= (4 * np.power(subst_h, 2) * np.sqrt(metal_cond))
    loss_conductor *= (np.power(beta * radius, 2) - np.power(n, 2))

    rad_conductance = np.power(beta_0 * radius, 2) / 480
    rad_conductance *= quad(calc_rad_cond_integral, 0, np.pi/2,
                            args=(n, beta_0, radius))[0]

    G_edge = rad_conductance + loss_dielectric + loss_conductor

    r0 = np.linspace(0, radius, num=101)

    zin_r = [np.power(jv(n, beta*r), 2) /
             (G_edge * np.power(jv(n, beta*radius), 2)) for r in
             r0]
    zin_r = np.array(zin_r)

    r_closest = np.abs(zin_r - zin_goal).argmin()
    r_closest = r0[r_closest]
    arg_init = beta * r_closest

    foo = zin_goal * np.power(jv(n, beta * radius), 2) * G_edge
    foo = np.sqrt(foo)

    arg_optim = newton(lambda x: jv(n, x) - foo, arg_init)
    r_optim = arg_optim / beta

    results_dict = {'q': result('Overall Q factor', Q),
                    'q_d': result('Dielectric Q factor', Q_dielectric),
                    'q_c': result('Conductor Q factor', Q_conductor),
                    'q_r': result('Radiation Q factor', Q_radiation),
                    'g_edge': result('Edge conductance', G_edge),
                    'r_optim': result('Distance from centre for 50 Ohm',
                                      r_optim)
                    }

    return results_dict


def calc_effect_dielectric_cover(subst_h, subst_er, cover_h, cover_er, air_er,
                                 patch_w, patch_l):
    er_mimic = subst_er
    er_multilayer = er_mimic

    for _ in range(2):

        effective_width = calc_effective_width(patch_w, subst_h, er_mimic,
                                               er_multilayer)
        ve = calc_ve(subst_h, cover_h, effective_width)
        q1 = calc_q1(subst_h, effective_width)
        q3 = calc_q3(subst_h, cover_h, effective_width, ve)
        q2 = calc_q2(q1, q3)
        q4 = calc_q4(subst_h, effective_width)

        q1 = q1 - q4
        q2 = 1 - q1 - q3 - 2 * q4

        er_multilayer = calc_er_multilayer(q1, q2, q3, q4, subst_er, cover_er,
                                           air_er)
        er_mimic = calc_er_mimic(er_multilayer, subst_h, effective_width)

    subst_er_eff = calc_er_eff(effective_width, er_mimic, subst_h)

    delta_patch_l_approx = calc_delta_l_approx(subst_er, subst_h,
                                               effective_width)
    delta_patch_l_exact = calc_delta_l_exact(er_mimic, subst_h,
                                             effective_width)

    freq_new = c0 / (2 * (patch_l + 2 * delta_patch_l_approx) *
                     np.sqrt(subst_er_eff))
    freq_new_exact = c0 / (2 * (patch_l + 2 * delta_patch_l_exact) *
                           np.sqrt(subst_er_eff))

    return (freq_new, freq_new_exact)


def calc_delta_l_approx(subst_er, subst_h, width):
    subst_er_eff = calc_er_eff(width, subst_er, subst_h)

    delta_patch_l = 0.412 * (subst_er_eff + 0.3) * (width / subst_h + 0.264)
    delta_patch_l /= ((subst_er_eff - 0.258) * (width / subst_h + 0.8))
    delta_patch_l *= subst_h

    return delta_patch_l


def calc_delta_l_exact(subst_er, subst_h, width):
    subst_er_eff = calc_er_eff(width, subst_er, subst_h)

    eta_5 = 1 - 0.218 * np.exp(-7.5 * width / subst_h)

    eta_2 = 1 + np.power(width / subst_h, 0.371) / (2.358 * subst_er + 1)

    eta_4_1 = 6 - 5 * np.exp(0.036 * (1 - subst_er))
    eta_4_2 = 0.067 * np.power(width / subst_h, 1.456)
    eta_4 = 1 + 0.0377 * np.arctan(eta_4_2) * eta_4_1

    eta_3_1 = 0.084 * np.power(width / subst_h, 1.9413 / eta_2)
    eta_3_2 = 0.5274 * np.arctan(eta_3_1) / np.power(subst_er_eff, 0.9236)
    eta_3 = 1 + eta_3_2

    eta_1_1 = np.power(subst_er_eff, 0.81)
    eta_1_2 = np.power(width / subst_h, 0.8544)
    eta_1_3 = (eta_1_1 + 0.26) / (eta_1_1 - 0.189)
    eta_1_4 = (eta_1_2 + 0.236) / (eta_1_2 + 0.87)
    eta_1 = 0.434907 * eta_1_3 * eta_1_4

    delta_l_exact = eta_1 * eta_3 * eta_5 / eta_4
    delta_l_exact = delta_l_exact * subst_h

    return delta_l_exact


def calc_effective_width(width, subst_h, er_mimic, er_multilayer):
    a = np.sqrt(er_mimic / er_multilayer)

    b = 0.164 * subst_h * (er_mimic - 1) / np.power(er_mimic, 2)
    part_a = a * (width + 0.882 * subst_h + b)

    c = subst_h * (er_mimic + 1) / (np.pi * er_mimic)
    d = np.log(width / (2 * subst_h) + 0.94) + 1.451
    part_b = a * c * d

    effective_width = part_a + part_b

    return effective_width


def calc_ve(subst_h, cover_h, effective_width):
    h2 = subst_h + cover_h
    arctan_arg = h2 / subst_h - 1
    arctan_arg *= np.pi
    arctan_arg /= (np.pi * effective_width / (2 * subst_h) - 2)

    ve = 2 * subst_h / np.pi * np.arctan(arctan_arg)

    return ve


def calc_er_mimic(er_multilayer, subst_h, effective_width):
    A = np.sqrt(1 + 10 * subst_h / effective_width)
    A = 1 / A

    er_mimic = (2 * er_multilayer - 1 + A) / (1 + A)

    return er_mimic


def calc_er_multilayer(q1, q2, q3, q4, subst_er, cover_er, air_er):
    a = subst_er * q1
    b = subst_er * np.power(1 - q1, 2)

    c = np.power(cover_er, 2) * q2 * q3
    d = cover_er * air_er
    e = q2 * q4 + np.power(q3 + q4, 2)

    f = np.power(cover_er, 2) * q2 * q3 * q4
    g = subst_er * (cover_er * q3 + air_er * q4) * np.power(1 - q1 - q4, 2)
    h = cover_er * air_er * q4
    i = q2 * q4 + np.power(q3 + q4, 2)

    j = f + g + h * i
    j = 1 / j

    er_multilayer = a + b * (c + d * e) * j

    return er_multilayer


def calc_q1(subst_h, effective_width):
    q1 = subst_h / (2 * effective_width)
    q1 *= np.log(np.pi * effective_width / subst_h - 1)
    q1 = 1 - q1

    return q1


def calc_q2(q1, q3):
    q2 = 1 - q1 - q3

    return q2


def calc_q3(subst_h, cover_h, effective_width, ve):
    h2 = subst_h + cover_h
    foo = ve * np.pi / (2 * subst_h)

    log_arg_1 = np.sin(foo)
    log_arg_2 = np.pi * effective_width / subst_h
    log_arg_3 = np.cos(foo)
    log_arg_4 = np.pi * (h2 / subst_h - 0.5) + foo

    log_arg = log_arg_2 * log_arg_3 / log_arg_4 + log_arg_1

    q3 = np.log(log_arg)
    q3 *= 0.5 * (subst_h - ve) / effective_width

    return q3


def calc_q4(subst_h, effective_width):
    foo = subst_h / (2 * effective_width)

    q4 = foo * np.log(np.pi / 2 - foo)

    return q4


def calc_line_impedance(patch_w, subst_er, subst_h):
    subst_er_eff = calc_er_eff(patch_w, subst_er, subst_h)

    if patch_w / subst_h <= 1:
        line_imp = 60 / np.sqrt(subst_er_eff)
        line_imp *= np.log(8 * subst_h / patch_w + patch_w / (4 * subst_h))
    else:
        line_imp_num = 120 * np.pi
        line_imp_denum_1 = np.sqrt(subst_er_eff)
        line_imp_denum_2 = patch_w / subst_h
        line_imp_denum_3 = 0.667 * np.log(patch_w / subst_h + 1.444)
        line_imp_denum = line_imp_denum_1 * (line_imp_denum_2 + 1.393 +
                                             line_imp_denum_3)
        line_imp = line_imp_num / line_imp_denum

    return line_imp


def calc_er_eff(patch_w, subst_er, subst_h):
    # effective relative permittivity using quasi-static approximation
    subst_er_eff = (subst_er - 1) / 2
    subst_er_eff *= np.power(1 + 12 * subst_h / patch_w, -0.5)
    subst_er_eff += (subst_er + 1) / 2

    return subst_er_eff


def patch_design_initial(freq, subst_er, subst_h):
    # initial guess for patch width
    patch_w = c0 / (2 * freq) * np.sqrt(2 / (subst_er + 1))

    subst_er_eff = calc_er_eff(patch_w, subst_er, subst_h)

    # effect of fringing fields on patch length
    delta_patch_l = calc_delta_l_approx(subst_er, subst_h, patch_w)

    # calculate patch length
    patch_l = c0 / (2 * freq * np.sqrt(subst_er_eff))
    patch_l -= (2 * delta_patch_l)

    results_dict = {'patch_w': result('Patch width', patch_w),
                    'patch_l': result('Patch length', patch_l),
                    'delta_patch_l':
                    result('Patch length extension due to fringing fields',
                           delta_patch_l)
                    }

    return results_dict


def patch_design_uslot(subst_h, subst_er, freq_centre, freq_bw):
    # design a patch with an U-slot according to procedure in AEH
    # freq_bw is in the same units as freq_centre
    # refer to AEH pg. 605
    freq_low = freq_centre - freq_bw / 2
    freq_high = freq_centre + freq_bw / 2
    lambda_centre = c0 / freq_centre

    if subst_h < (0.06 * lambda_centre / np.sqrt(subst_er)):
        print('Broadband operation unlikely')

    patch_l_ext = c0 / (2 * np.sqrt(subst_er) * freq_centre)
    patch_w = 1.5 * patch_l_ext

    subst_er_eff = calc_er_eff(patch_w, subst_er, subst_h)

    delta_patch_l = calc_delta_l_approx(subst_er, subst_h, patch_w)
    delta_patch_l *= 2

    patch_l = c0 / (2 * np.sqrt(subst_er_eff) * freq_centre) - delta_patch_l

    gap_width = lambda_centre / 60

    slot_width = c0 / (np.sqrt(subst_er_eff) * freq_low) - 2 * (patch_l +
                                                                delta_patch_l -
                                                                gap_width)

    slot_length = np.max([0.3 * patch_w, 0.75 * slot_width])

    ppatch_w = slot_width - 2 * gap_width  # pseudo patch

    subst_er_eff_pp = calc_er_eff(ppatch_w, subst_er, subst_h)

    delta_ppatch_l = calc_delta_l_approx(subst_er, subst_h, ppatch_w)
    delta_ppatch_l *= 2

    slot_edge_offset = patch_l - gap_width + delta_ppatch_l
    foo = 1 / np.sqrt(subst_er_eff_pp)
    foo *= (c0 / freq_high - (2 * slot_length + slot_width))
    slot_edge_offset = slot_edge_offset - foo

    results_dict = {'patch_w': result('Patch width', patch_w),
                    'patch_l': result('Patch length', patch_l),
                    'delta_patch_l': result('Main patch L extension',
                                            delta_patch_l),
                    'delta_ppatch_l': result('Pseudopatch L extension',
                                             delta_ppatch_l),
                    'slot_width': result('Length of slot along patch width',
                                         slot_width),
                    'slot_length': result('Length of slot along patch length',
                                          slot_length),
                    'gap_width': result('Slot cutout width', gap_width),
                    'slot_edge_offset':
                    result('Distance between U slot and radiating patch edge',
                           slot_edge_offset)
                    }

    return results_dict


def calc_trim_length_rect_patch(patch_l, Q):
    # calculates corner trim length for a square patch needed to achieve CP
    delta_l = patch_l / np.sqrt(Q)

    results_dict = {'delta_l': result('Trim length', delta_l)}

    return results_dict


def calc_edge_offset(r_edge, r_goal, patch_l):
    # calculates offset from radiation edge for a goal input resistance
    offset = r_goal / r_edge
    offset_2 = np.sqrt(offset)
    offset_4 = np.power(offset, 1/4)

    offset_2 = np.arccos(offset_2)
    offset_4 = np.arccos(offset_4)

    offset_2 = offset_2 * patch_l / np.pi
    offset_4 = offset_4 * patch_l / np.pi

    results_dict = {'offset_2': result('Offset using cos^2', offset_2),
                    'offset_4': result('Offset using cos^4', offset_4)
                    }

    return results_dict


def calc_patch_bandwidths(VSWR, Q, subst_er, subst_h, lambda_0,
                          patch_w=None, patch_l=None):
    # calculate the expected bandwidth of the designed patch antenna
    BW_linear = (VSWR - 1) / (Q * np.sqrt(VSWR))
    BW_opt = (1 / (2 * Q)) * np.sqrt(np.power(VSWR, 4) - 1) / VSWR
    BW_max = (1 / Q) * np.pi / np.log((VSWR + 1) / (VSWR - 1))

    if patch_w is not None and patch_l is not None:
        BW_check = 3.771 * (subst_er - 1) / np.power(subst_er, 2)
        BW_check *= (subst_h / lambda_0) * (patch_w / patch_l)
    else:
        BW_check = BW_linear

    results_dict = {'BW_linear': result('Linear polarisation BW', BW_linear),
                    'BW_opt': result('Optimal mismatch BW', BW_opt),
                    'BW_max': result('Maximum BW', BW_max),
                    'BW_check': result('Linear polarisation BW check',
                                       BW_check)
                    }

    return results_dict


def calc_zin_q_atma(freq, lambda_0, subst_h, subst_er, subst_tand, metal_cond,
                    metal_ur, patch_l, patch_w, delta_patch_l):
    # uses method from Antenna Theory and Microstrip Antennas
    subst_er_eff = calc_er_eff(patch_w, subst_er, subst_h)
    patch_Z0 = calc_line_impedance(patch_w, subst_er, subst_h)
    patch_Y0 = 1 / patch_Z0

    lambda_g = lambda_0 / np.sqrt(subst_er_eff)
    beta_g = 2 * np.pi / lambda_g

    if patch_w < 0.35 * lambda_0:
        Gs = (1 / 90) * np.power(patch_w / lambda_0, 2)
    elif (patch_w >= 0.35 * lambda_0) and (patch_w < 2 * lambda_0):
        Gs = (1 / 120) * patch_w / lambda_0 - (1 / (60 * np.power(np.pi, 2)))
    else:
        Gs = patch_w / (120 * lambda_0)

    Gr = 2 * Gs

    tan = np.tan(beta_g * (patch_l + 2 * delta_patch_l))
    patch_Yin = Gs + complex(0, patch_Y0 * tan)
    patch_Yin /= (patch_Y0 + complex(0, Gs * tan))
    patch_Yin *= patch_Y0
    patch_Yin += Gs
    patch_Zin = 1 / patch_Yin

    Q_dielectric = 1 / subst_tand

    Q_conductor = (1 / np.pi) * np.sqrt(lambda_0 / (120 * metal_cond))
    Q_conductor = subst_h / Q_conductor

    Q_spacewave = subst_er * patch_w * patch_l
    Q_spacewave /= (60 * lambda_0 * subst_h * 1 * 2 * Gr)

    He = subst_h / lambda_0 * np.sqrt(subst_er - 1)  # might be lambda_0

    if subst_h / lambda_g < 0.06:
        eta_s = 1 - 3.4 * He
    elif (subst_h / lambda_g > 0.06) and (subst_h / lambda_g < 0.16):
        eta_s = 1 - 3.4 * He
        eta_s += (1600 / np.power(subst_er, 3) * (np.power(He, 3) - 100 *
                                                  np.power(He, 5.6)))
    else:
        print("Substrate too thick.")

    Q_surfacewave = Q_spacewave * (eta_s / (1 - eta_s))

    # Overall Q factor
    Q = 1 / Q_dielectric + 1 / Q_conductor + 1 / Q_spacewave + \
        1 / Q_surfacewave
    Q = 1 / Q

    Q_check = 120 * lambda_0 * subst_h * Gr
    Q_check /= (subst_er * patch_w * patch_l * (1 - 3.4 * He))
    Q_check += subst_tand
    Q_check += (1 / (np.pi * subst_h) * np.sqrt(lambda_0 / (120 * metal_cond)))
    Q_check = 1 / Q_check

    a = patch_w + 2 * delta_patch_l
    b = patch_l + 2 * delta_patch_l
    lambda_01 = 2 * b * np.sqrt(subst_er_eff)
    patch_R01 = 120 * lambda_01 * Q * subst_h / (subst_er * a * b)

    results_dict = {'book': result('Design procedure accroding to',
                                   'Antenna Theory and Microstrip Antennas'),
                    'q_d': result('Dielectric Q factor', Q_dielectric),
                    'q_c': result('Conductor Q factor', Q_conductor),
                    'q_r': result('Radiation Q factor', Q_spacewave),
                    'q_sw': result('Surfacewave Q factor', Q_surfacewave),
                    'q': result('Overall Q factor', [Q, Q_check]),
                    'z_in': result('Edge impedance',
                                   [1/Gr, patch_Zin, patch_R01])
                    }

    return results_dict


def calc_zin_q_aeh(freq, lambda_0, subst_h, subst_er, subst_ur, subst_tand,
                   metal_cond, metal_ur, patch_l, patch_w, delta_patch_l):
    # uses method from Antenna Engineering Handbook
    beta_0 = 2 * np.pi / lambda_0

    # Losses due to dielectric
    Q_dielectric = 1 / subst_tand

    # Losses in the conductor
    skin_depth = np.sqrt(2 / (2 * np.pi * freq * mu_0 * metal_cond * metal_ur))
    surface_resistance = 1 / (skin_depth * metal_cond)
    Q_conductor = metal_ur * (eta_0 / 2) * (beta_0 * subst_h /
                                            surface_resistance)

    # Radiated wave Q factor
    delta_patch_w = subst_h * np.log(4) / np.pi
    Le = patch_l + 2 * delta_patch_l
    We = patch_w + 2 * delta_patch_w

    a2 = -0.16605
    a4 = 0.00761
    c2 = -0.0914153
    n1 = np.sqrt(subst_er * subst_ur)

    c1 = 1 - 1 / np.power(n1, 2) + (2 / 5) / np.power(n1, 4)

    pr = 1 + (a2 / 10) * np.power(beta_0 * We, 2)
    pr += (np.power(a2, 2) + 2 * a4) * (3 / 560) * np.power(beta_0 * We, 4)
    pr += c2 * (1 / 5) * np.power(beta_0 * Le, 2)
    pr += a2 * c2 * (1 / 70) * np.power(beta_0 * We * beta_0 * Le, 2)

    Q_spacewave = (3 / 16) * (subst_er / (pr * c1)) * (Le / We) * (lambda_0 /
                                                                   subst_h)

    # Surface wave Q factor
    s = np.sqrt(subst_er - 1)
    arg_trig = beta_0 * subst_h * s

    alpha_0 = s * np.tan(arg_trig)

    alpha_1 = np.tan(arg_trig)
    alpha_1 += (arg_trig / np.power(np.cos(arg_trig), 2))
    alpha_1 *= (1 / s)
    alpha_1 = -alpha_1

    x0 = alpha_0 * alpha_1 - np.power(subst_er, 2)
    x0_sqrt = np.power(subst_er, 2) - 2*alpha_0*alpha_1 + np.power(alpha_0, 2)
    x0 += subst_er * np.sqrt(x0_sqrt)
    x0 /= (np.power(subst_er, 2) - np.power(alpha_1, 2))
    x0 += 1

    x1 = (np.power(x0, 2) - 1) / (subst_er - np.power(x0, 2))

    P_sw = eta_0 * np.power(beta_0, 2) / 4
    P_sw *= (subst_er * np.power(np.power(x0, 2) - 1, 3/2))

    P_sw_denum = (beta_0 * subst_h) * np.sqrt(np.power(x0, 2) - 1)
    P_sw_denum *= (1 + x1 * np.power(subst_er, 2))
    P_sw_denum += (subst_er * (1 + x1))

    P_sw = P_sw / P_sw_denum

    P_sp = 1 / np.power(lambda_0, 2)
    P_sp *= np.power(beta_0 * subst_h, 2)
    P_sp *= (80 * np.power(np.pi, 2) * np.power(metal_ur, 2) * c1)

    e_sw = P_sp / (P_sp + P_sw)

    Q_surfacewave = Q_spacewave * (e_sw / (1 - e_sw))

    P_sw = 1 / np.power(lambda_0, 2)
    P_sw *= np.power(beta_0 * subst_h, 3)
    P_sw *= (60 * np.power(np.pi * subst_ur, 3))
    P_sw *= np.power(1 - 1 / np.power(n1, 2), 3)

    e_sw = P_sp / (P_sp + P_sw)
    Q_surfacewave_check = Q_spacewave * (e_sw / (1 - e_sw))

    # Overall Q factor
    Q = 1 / Q_dielectric + 1 / Q_conductor + 1 / Q_spacewave + \
        1 / Q_surfacewave
    Q = 1 / Q
    Q_check = 1 / Q_dielectric + 1 / Q_conductor
    Q_check += 1 / Q_spacewave + 1 / Q_surfacewave_check
    Q_check = 1 / Q_check

    R_edge = eta_0 * subst_ur * (4 / np.pi) * (Le / We) * (subst_h / lambda_0)
    R_edge_check = R_edge * Q_check
    R_edge = R_edge * Q

    results_dict = {'book': result('Design procedure accroding to',
                                   'Antenna Engineering Handbook'),
                    'q_d': result('Dielectric Q factor', Q_dielectric),
                    'q_c': result('Conductor Q factor', Q_conductor),
                    'q_r': result('Radiation Q factor', Q_spacewave),
                    'q_sw': result('Surfacewave Q factor',
                                   [Q_surfacewave, Q_surfacewave_check]),
                    'q': result('Overall Q factor', [Q, Q_check]),
                    'z_in': result('Edge impedance',
                                   [R_edge, R_edge_check])
                    }

    return results_dict


def calc_zin_q_mpad(freq, lambda_0, subst_h, subst_er, subst_ur, subst_tand,
                    metal_cond, metal_ur, patch_l, patch_w, delta_patch_l):
    # uses method from Microstrip and Printed Antenna Design
    subst_er_eff = calc_er_eff(patch_w, subst_er, subst_h)
    patch_Z0 = calc_line_impedance(patch_w, subst_er, subst_h)
    patch_Y0 = 1 / patch_Z0

    lambda_g = lambda_0 / np.sqrt(subst_er_eff)
    beta_0 = 2 * np.pi / lambda_0
    beta_g = 2 * np.pi / lambda_g

    Ge = 0.00836 * patch_w / lambda_0
    Be = 0.01668 * delta_patch_l * patch_w * subst_er_eff / \
        (subst_h * lambda_0)
    Ye = complex(Ge, Be)

    tan = np.tan(beta_g * patch_l)
    patch_Yin = Ye + complex(0, patch_Y0 * tan)
    patch_Yin /= (patch_Y0 + complex(0, Ye * tan))
    patch_Yin *= patch_Y0
    patch_Yin += Ye
    patch_Zin = 1 / patch_Yin

    Q_dielectric = 1 / subst_tand

    Rs = np.sqrt(2 * np.pi * freq * mu_0 / (2 * metal_cond))
    Q_conductor = 0.5 * eta_0 * metal_ur * beta_0 * subst_h / Rs

    A = np.power(np.pi * patch_w / lambda_0, 2)
    B = np.power(2 * patch_l / lambda_0, 2)

    Pr = A * np.power(np.pi, 4) / 23040
    Pr_1 = (1 - B)
    Pr_1 *= (1 - A / 15 + np.power(A, 2) / 420)
    Pr_2 = np.power(B, 2) / 5
    Pr_2 *= (2 - A / 7 + np.power(A, 2) / 189)
    Pr = Pr * (Pr_1 + Pr_2)

    Wes = epsilon_0 * subst_er * patch_l * patch_w / (8 * subst_h)

    Q_spacewave = 2 * 2 * np.pi * freq * Wes / Pr

    n1 = np.sqrt(subst_er * subst_ur)
    c1 = 1 - 1 / np.power(n1, 2) + 2 / (5 * np.power(n1, 4))

    Psp = np.power(beta_0 * subst_h, 2) / np.power(lambda_0, 2)
    Psp *= 80 * c1 * np.power(np.pi * subst_ur, 2)

    alpha_0 = np.sqrt(subst_er - 1) * np.tan(beta_0 * subst_h *
                                             np.sqrt(subst_er - 1))
    alpha_1 = np.tan(beta_0 * subst_h * np.sqrt(subst_er - 1))
    alpha_1_1 = beta_0 * subst_h * np.sqrt(subst_er - 1)
    alpha_1_2 = np.cos(beta_0 * subst_h * np.sqrt(subst_er - 1))
    alpha_1_2 = np.power(alpha_1_2, 2)
    alpha_1 += alpha_1_1 / alpha_1_2
    alpha_1 /= np.sqrt(subst_er - 1)
    alpha_1 = -alpha_1

    x0 = alpha_0 * alpha_1 - np.power(subst_er, 2)
    x0_sqrt = np.power(subst_er, 2) - 2*alpha_0*alpha_1 + np.power(alpha_0, 2)
    x0_sqrt = np.sqrt(x0_sqrt)
    x0 += subst_er * x0_sqrt
    x0 /= (np.power(subst_er, 2) - np.power(alpha_1, 2))
    x0 += 1

    x1 = np.power(x0, 2) - 1
    x1 /= (subst_er - np.power(x0, 2))

    Psw = eta_0 * np.power(beta_0, 2) / 8
    Psw *= subst_er * np.power(np.power(x0, 2) - 1, 1.5)

    Psw_denum_1 = subst_er * (1 + x1)
    Psw_denum_2 = beta_0 * subst_h * np.sqrt(np.power(x0, 2) - 1)
    Psw_denum_2 *= (1 + np.power(subst_er, 2) * x1)
    Psw_denum = Psw_denum_1 + Psw_denum_2

    Psw = Psw / Psw_denum

    er = Psp / (Psp + Psw)

    Q_surfacewave = Q_spacewave * (er / (1 - er))

    Q = 1 / Q_dielectric + 1 / Q_conductor + 1 / Q_spacewave + \
        1 / Q_surfacewave
    Q = 1 / Q

    results_dict = {'book': result('Design procedure accroding to',
                                   'Microstrip and Printed Antenna Design'),
                    'q_d': result('Dielectric Q factor', Q_dielectric),
                    'q_c': result('Conductor Q factor', Q_conductor),
                    'q_r': result('Radiation Q factor', Q_spacewave),
                    'q_sw': result('Surfacewave Q factor', Q_surfacewave),
                    'q': result('Overall Q factor', Q),
                    'z_in': result('Edge impedance',
                                   [patch_Zin, 1/(2*Ge)])
                    }

    return results_dict


def calc_zin_q_atad(freq, lambda_0, subst_h, subst_er, subst_tand, metal_cond,
                    metal_ur, patch_l, patch_w, delta_patch_l):
    # uses method from Antenna Theory, Analysis, and Design
    subst_er_eff = calc_er_eff(patch_w, subst_er, subst_h)
    patch_Z0 = calc_line_impedance(patch_w, subst_er, subst_h)
    patch_Y0 = 1 / patch_Z0

    lambda_g = lambda_0 / np.sqrt(subst_er_eff)
    beta_0 = 2 * np.pi / lambda_0
    beta_g = 2 * np.pi / lambda_g

    G1 = 1 / 24.0
    G1 = G1 * np.power(beta_0 * subst_h, 2)
    G1 = 1 - G1
    G1 = patch_w * G1
    G1 = G1 / (120 * lambda_0)

    B1 = 1 - 0.636 * np.log(beta_0 * subst_h)
    B1 *= patch_w / (120 * lambda_0)

    Y1 = complex(G1, B1)

    tan = np.tan(beta_g * patch_l)
    patch_Yin = Y1 + complex(0, patch_Y0 * tan)
    patch_Yin /= (patch_Y0 + complex(0, Y1 * tan))
    patch_Yin *= patch_Y0
    patch_Yin += Y1

    patch_Zin = 1 / (2 * G1)

    patch_Zin_check = 90 * np.power(subst_er, 2) / (subst_er - 1)
    patch_Zin_check = patch_Zin_check * (patch_l / patch_w)

    Q_dielectric = 1 / subst_tand

    Q_conductor = subst_h * np.sqrt(np.pi * freq * metal_cond *
                                    metal_ur * mu_0)

    Q_spacewave = 2 * 2 * np.pi * freq * subst_er * epsilon_0 * patch_l / 4
    Q_spacewave /= (subst_h * 2 * G1 / patch_w)

    Q = 1 / Q_dielectric + 1 / Q_conductor + 1 / Q_spacewave
    Q = 1 / Q

    results_dict = {'book': result('Design procedure accroding to',
                                   'Antenna Theory, Analysis, and Design'),
                    'q_d': result('Dielectric Q factor', Q_dielectric),
                    'q_c': result('Conductor Q factor', Q_conductor),
                    'q_r': result('Radiation Q factor', Q_spacewave),
                    'q_sw': result('Surfacewave Q factor', 'N/A'),
                    'q': result('Overall Q factor', Q),
                    'z_in': result('Edge impedance',
                                   [1/(2*G1), patch_Zin, patch_Zin_check,
                                    1/patch_Yin])
                    }

    return results_dict
