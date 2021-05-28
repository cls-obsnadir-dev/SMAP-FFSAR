#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pyproj

class Cst():
    """
    Constants and useful variables for processing S3 data
    """
    def __init__(self):

        # Constants
        
        self.c=299792458.0

        self.n_sample_range=128
        self.n_pulse_burst=64

        self.fc=13.575e9
        self.B=320e6
        self.Fs=320e6
        self.T=44.8e-6
        self.alpha=self.B/self.T
        self.BRI=1018710*12.5e-9
        self.PRI=4488*12.5e-9
        self.theta3dB=np.radians(1.34)
        self.gamma=np.sin(self.theta3dB)**2 / (2*np.log(2))
        self.abs_ref_track=44
        self.GdB=41.9 # antenna gain in dB

        self.time_tol=self.PRI

        self.tau_offset=77.576/self.B
        self.tracker_phase_shift=2.567
        
        self.sig0_bias_ocean=-2.28
        self.sig0_bias_ocean_lrm=-2.28
        
        self.sig0_bias_ocog_sar=11.51
        self.sig0_bias_ocog_lrm=0.0
        
        # Coefficients of the hamming window
        self.a1=0.5
        self.a2=0.5

        # Variables

        self.ecef=pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
        self.lla=pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
        self.geoid_wgs=pyproj.Geod(ellps="WGS84")

        self.t=np.arange(-self.n_sample_range//2,self.n_sample_range//2)/self.n_sample_range*self.T
        self.tau=np.arange(-self.n_sample_range//2,self.n_sample_range//2)/self.B
        self.eta_burst=np.arange(-self.n_pulse_burst//2,self.n_pulse_burst//2)*self.PRI

        

        
