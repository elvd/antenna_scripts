# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 12:40:33 2017

Quick calculation of centre and lower frequency for a TRL calibration kit,
given the higher frequency and the phase span of the Line standard.

@author: eenvdo
"""

freq_high = 6.0  # in arbitrary units
span = 160.0  # in degrees

freq_centre = freq_high * (90 / (90 + float(span) / 2))
freq_low = freq_centre * ((90 - float(span)/2) / 90)

print(freq_centre)
print(freq_low)
