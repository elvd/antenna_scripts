"""
Extracts data for complex relative permittivity of a dielectric substrate
from S-parameter measurements of two transmission lines of different lenghts.

@author: eenvdo
"""

import numpy as np
from scipy.constants import speed_of_light as c0


def calc_tan_delta(complex_er):
    # Calculates loss tangent from complex relative permittivity
    # as er_imag / er_real
    tan_delta = [er.imag / er.real for er in complex_er]

    return tan_delta


def complex_er_from_abcd(delta_d, sparams_long, sparams_short):
    # Uses ABCD matrices. Inputs are scikit-rf objects.
    assert np.shape(sparams_long.s) == np.shape(sparams_short.s), \
           'Different number of measurement points'

    complex_er = np.empty(np.shape(sparams_long.s)[0], dtype=complex)
    error_connectors = 0

    for idx, freq_point in enumerate(sparams_long.frequency.f):
        beta_0 = 2 * np.pi * freq_point / c0

        a_short = sparams_short[idx].a[0]
        a_long = sparams_long[idx].a[0]

        dot_product = np.dot(a_short, np.linalg.inv(a_long))
        gamma = np.arccosh(0.5 * np.trace(dot_product)) / delta_d

        alpha = gamma.real
        beta = gamma.imag

        real_er = (np.power(beta, 2) - np.power(alpha, 2))
        real_er /= np.power(beta_0, 2)
        imag_er = (-2 * alpha * beta) / np.power(beta_0, 2)

        complex_er[idx] = complex(real_er, imag_er)

        trace_1 = np.trace(np.dot(a_short, np.linalg.inv(a_long)))
        trace_2 = np.trace(np.dot(a_long, np.linalg.inv(a_short)))
        trace_3 = np.trace(np.dot(np.linalg.inv(a_short), a_long))
        trace_4 = np.trace(np.dot(np.linalg.inv(a_long), a_short))
        point_error = np.abs(trace_1 - trace_2) + np.abs(trace_3 - trace_4)
        error_connectors = error_connectors + point_error

    error_connectors = error_connectors / np.size(sparams_long.frequency.f)

    return (complex_er, error_connectors)


def complex_er_from_t(delta_d, sparams_long, sparams_short):
    # Uses T matrices. Inputs are scikit-rf objects.
    assert np.shape(sparams_long.s) == np.shape(sparams_short.s), \
           'Different number of measurement points'

    complex_er = np.empty(np.shape(sparams_long.s)[0], dtype=complex)

    for idx, freq_point in enumerate(sparams_long.frequency.f):
        beta_0 = 2 * np.pi * freq_point / c0

        t_short = sparams_short[idx].t[0]
        t_long = sparams_long[idx].t[0]

        u = np.dot(t_short, np.linalg.inv(t_long))
        u += np.dot(t_long, np.linalg.inv(t_short))
        u = np.trace(u) / 2

        eigenvalue = (u + np.sqrt(np.power(u, 2) - 4)) / 2
        gamma = np.log(eigenvalue) / delta_d

        alpha = gamma.real
        beta = gamma.imag

        real_er = (np.power(beta, 2) - np.power(alpha, 2))
        real_er /= np.power(beta_0, 2)
        imag_er = (-2 * alpha * beta) / np.power(beta_0, 2)

        complex_er[idx] = complex(real_er, imag_er)

    return complex_er


def complex_er_from_t2(delta_d, sparams_long, sparams_short):
    # Uses T matrices. Inputs are scikit-rf objects. Alternative method
    assert np.shape(sparams_long.s) == np.shape(sparams_short.s), \
           'Different number of measurement points'

    complex_er = np.empty(np.shape(sparams_long.s)[0], dtype=complex)

    for idx, freq_point in enumerate(sparams_long.frequency.f):
        beta_0 = 2 * np.pi * freq_point / c0

        t_short = sparams_short[idx].t[0]
        t_long = sparams_long[idx].t[0]

        u = 2 - np.linalg.det(t_short + t_long) / np.linalg.det(t_long)
        eigenvalue = (-u + np.sqrt(np.power(u, 2) - 4)) / 2
        gamma = np.log(eigenvalue) / delta_d

        alpha = gamma.real
        beta = gamma.imag

        real_er = (np.power(beta, 2) - np.power(alpha, 2))
        real_er /= np.power(beta_0, 2)
        imag_er = (-2 * alpha * beta) / np.power(beta_0, 2)

        complex_er[idx] = complex(real_er, imag_er)

    return complex_er
