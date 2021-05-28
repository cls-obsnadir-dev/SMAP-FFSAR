#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import logging

from cst import Cst

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

cst=Cst()

class DataBuffer:
    
    """
    Object containing a portion of raw L1a data. The main interest of using this class is to read once a big chunk of data from a L1A netCDF file, and to keep it in memory. In fact, accessing the data from the netCDF file burst by burst is highly inefficient because of the chunking of the L1A files.
    """

    def __init__(self,file_l1a,n_burst):

        """
        Constructor of the class.
        """

        self.file_l1a=file_l1a
        self.n_burst=n_burst
        self.n_bursts_l1a=self.file_l1a.dimensions['time_l1a_echo_sar_ku'].size

        self.index_start=np.nan
        self.index_end=np.nan
                
        self.time=np.empty(self.n_burst,dtype=np.float64)
        self.lat=np.empty(self.n_burst,dtype=np.float64)
        self.lon=np.empty(self.n_burst,dtype=np.float64)
        self.surf_type=np.empty(self.n_burst,dtype=np.int8)
        self.xs=np.empty(self.n_burst,dtype=np.float64)
        self.ys=np.empty(self.n_burst,dtype=np.float64)
        self.zs=np.empty(self.n_burst,dtype=np.float64)
        self.vxs=np.empty(self.n_burst,dtype=np.float64)
        self.vys=np.empty(self.n_burst,dtype=np.float64)
        self.vzs=np.empty(self.n_burst,dtype=np.float64)
        self.alt=np.empty(self.n_burst,dtype=np.float64)
        self.orb_alt_rate=np.empty(self.n_burst,dtype=np.float64)
        self.h0_applied=np.empty(self.n_burst,dtype=np.uint32)
        self.cor2_applied=np.empty(self.n_burst,dtype=np.int16)
        self.tracker=np.empty(self.n_burst,dtype=np.float64)
        self.cog=np.empty(self.n_burst,dtype=np.float64)
        self.agc=np.empty(self.n_burst,dtype=np.float64)
        self.scale_factor=np.empty(self.n_burst,dtype=np.float64)
        self.sig0_cal=np.empty(self.n_burst,dtype=np.float64)
        self.i_count=np.empty((self.n_burst,cst.n_pulse_burst,cst.n_sample_range),dtype=np.int8)
        self.q_count=np.empty((self.n_burst,cst.n_pulse_burst,cst.n_sample_range),dtype=np.int8)
        self.burst_count_cycle=np.empty(self.n_burst,dtype=np.int8)

        self.burst_power=np.empty(cst.n_pulse_burst,dtype=np.float64)
        self.burst_phase=np.empty(cst.n_pulse_burst,dtype=np.float64)

    def load(self,index_start,index_end):

        """
        Load bursts from index_start to index_end. index_start must greater or equal to zero. index_end must be greater than index_start. If index_end - index_start < n_burst, NaNs are put in place of data. If index_end is greater than the dimension of the l1a file, NaNs are put in place of the data.
        """

        logging.debug("Loading data buffer from indexes {} to {}.".format(index_start,index_end))

        self.index_start=index_start
        self.index_end=index_end

        try:
            assert(self.index_start>=0 and self.index_start<self.n_bursts_l1a and self.index_end>self.index_start)
        except AssertionError:
            logging.error("index_start must be greater than or equal to zero and lesser than the size of the l1a file. index_end must be greater than index_start.")
        else:
            indices_ok=True

        if indices_ok:

            index_max=min(self.n_bursts_l1a,self.index_end)

            # Reading the data from index_start to index_max
            self.time[:index_max-index_start]=self.file_l1a["time_l1a_echo_sar_ku"][index_start:index_max]
            self.lat[:index_max-index_start]=self.file_l1a["lat_l1a_echo_sar_ku"][index_start:index_max]
            self.lon[:index_max-index_start]=self.file_l1a["lon_l1a_echo_sar_ku"][index_start:index_max]
            self.surf_type[:index_max-index_start]=self.file_l1a["surf_type_l1a_echo_sar_ku"][index_start:index_max]      
            self.xs[:index_max-index_start]=self.file_l1a["x_pos_l1a_echo_sar_ku"][index_start:index_max]
            self.ys[:index_max-index_start]=self.file_l1a["y_pos_l1a_echo_sar_ku"][index_start:index_max]
            self.zs[:index_max-index_start]=self.file_l1a["z_pos_l1a_echo_sar_ku"][index_start:index_max]
            self.vxs[:index_max-index_start]=self.file_l1a["x_vel_l1a_echo_sar_ku"][index_start:index_max]
            self.vys[:index_max-index_start]=self.file_l1a["y_vel_l1a_echo_sar_ku"][index_start:index_max]
            self.vzs[:index_max-index_start]=self.file_l1a["z_vel_l1a_echo_sar_ku"][index_start:index_max]
            self.alt[:index_max-index_start]=self.file_l1a["alt_l1a_echo_sar_ku"][index_start:index_max]
            self.orb_alt_rate[:index_max-index_start]=self.file_l1a["orb_alt_rate_l1a_echo_sar_ku"][index_start:index_max]
            self.h0_applied[:index_max-index_start]=self.file_l1a["h0_applied_l1a_echo_sar_ku"][index_start:index_max]
            self.cor2_applied[:index_max-index_start]=self.file_l1a["cor2_applied_l1a_echo_sar_ku"][index_start:index_max]
            self.tracker[:index_max-index_start]=self.file_l1a["range_ku_l1a_echo_sar_ku"][index_start:index_max]
            self.cog[:index_max-index_start]=self.file_l1a["cog_cor_l1a_echo_sar_ku"][index_start:index_max]
            self.agc[:index_max-index_start]=self.file_l1a["agc_ku_l1a_echo_sar_ku"][index_start:index_max]
            self.scale_factor[:index_max-index_start]=self.file_l1a["scale_factor_ku_l1a_echo_sar_ku"][index_start:index_max]
            self.sig0_cal[:index_max-index_start]=self.file_l1a["sig0_cal_ku_l1a_echo_sar_ku"][index_start:index_max]
            self.i_count[:index_max-index_start]=self.file_l1a["i_meas_ku_l1a_echo_sar_ku"][index_start:index_max]
            self.q_count[:index_max-index_start]=self.file_l1a["q_meas_ku_l1a_echo_sar_ku"][index_start:index_max]
            self.burst_count_cycle[:index_max-index_start]=self.file_l1a["burst_count_cycle_l1a_echo_sar_ku"][index_start:index_max]

            self.burst_power[:]=self.file_l1a["burst_power_cor_ku_l1a_echo_sar_ku"][index_start]
            self.burst_phase[:]=self.file_l1a["burst_phase_cor_ku_l1a_echo_sar_ku"][index_start]

            # Setting to nan/0 remaining bursts
            self.time[index_max-index_start:]=np.nan
            self.lat[index_max-index_start:]=0
            self.lon[index_max-index_start:]=0
            self.surf_type[index_max-index_start:]=0
            self.xs[index_max-index_start:]=0
            self.ys[index_max-index_start:]=0
            self.zs[index_max-index_start:]=0
            self.vxs[index_max-index_start:]=0
            self.vys[index_max-index_start:]=0
            self.vzs[index_max-index_start:]=0
            self.alt[index_max-index_start:]=0
            self.orb_alt_rate[index_max-index_start:]=0
            self.h0_applied[index_max-index_start:]=0
            self.cor2_applied[index_max-index_start:]=0
            self.tracker[index_max-index_start:]=0
            self.cog[index_max-index_start:]=0
            self.agc[index_max-index_start:]=0
            self.scale_factor[index_max-index_start:]=0
            self.sig0_cal[index_max-index_start:]=0
            self.i_count[index_max-index_start:]=0
            self.q_count[index_max-index_start:]=0
            self.burst_count_cycle[index_max-index_start:]=0

        else:

            # Setting to nan/0 if indices are not correct
            self.time[:]=np.nan
            self.lat[:]=0
            self.lon[:]=0
            self.surf_type[:]=0
            self.xs[:]=0
            self.ys[:]=0
            self.zs[:]=0
            self.vxs[:]=0
            self.vys[:]=0
            self.vzs[:]=0
            self.alt[:]=0
            self.orb_alt_rate[:]=0
            self.h0_applied[:]=0
            self.cor2_applied[:]=0
            self.tracker[:]=0
            self.cog[:]=0
            self.agc[:]=0
            self.scale_factor[:]=0
            self.sig0_cal[:]=0
            self.i_count[:]=0
            self.q_count[:]=0
            self.burst_count_cycle[:]=0
            
class DataBlock:
    
    """
    Object containing all necessary calibrated data to perform SAR processing. The size of the block is n_burst bursts. The calibrations (phase,power) are performed upon loading the data. The echoes are kept in the frequency domain. Setting a new data block is done by shifting the old one to prevent unnecessary computations.
    """
    
    def __init__(self,n_burst,data_buffer,gprw_mean):
                
        """
        Constructor of the class
        """

        # Block position and size
        self.n_burst = n_burst
        self.index_center = np.nan

        # Input data
        self.data_buffer = data_buffer
        self.gprw_mean = gprw_mean

        # Input data
        self.time=np.zeros(n_burst,dtype=np.float64)
        self.lat=np.zeros(n_burst,dtype=np.float64)
        self.lon=np.zeros(n_burst,dtype=np.float64)
        self.surf_type=np.zeros(n_burst,dtype=np.int8)
        self.xs=np.zeros(n_burst,dtype=np.float64)
        self.ys=np.zeros(n_burst,dtype=np.float64)
        self.zs=np.zeros(n_burst,dtype=np.float64)
        self.vxs=np.zeros(n_burst,dtype=np.float64)
        self.vys=np.zeros(n_burst,dtype=np.float64)
        self.vzs=np.zeros(n_burst,dtype=np.float64)
        self.alt=np.zeros(n_burst,dtype=np.float64)
        self.orb_alt_rate=np.zeros(n_burst,dtype=np.float64)
        self.h0_applied=np.zeros(n_burst,dtype=np.uint32)
        self.cor2_applied=np.zeros(n_burst,dtype=np.int16)
        self.tracker=np.zeros(n_burst,dtype=np.float64)
        self.cog=np.zeros(n_burst,dtype=np.float64)
        self.agc=np.zeros(n_burst,dtype=np.float64)
        self.scale_factor=np.zeros(n_burst,dtype=np.float64)
        self.sig0_cal=np.zeros(n_burst,dtype=np.float64)
        self.i_count=np.zeros((n_burst,cst.n_pulse_burst,cst.n_sample_range),dtype=np.int8)
        self.q_count=np.zeros((n_burst,cst.n_pulse_burst,cst.n_sample_range),dtype=np.int8)
        self.burst_count_cycle=np.zeros(n_burst,dtype=np.int8)       
        self.burst_power=np.zeros(cst.n_sample_range,dtype=np.float64)
        self.burst_phase=np.zeros(cst.n_sample_range,dtype=np.float64)

        # New data
        self.Echo=np.zeros((n_burst,cst.n_pulse_burst,cst.n_sample_range),dtype=np.complex64)
        self.echo=np.zeros((n_burst,cst.n_pulse_burst,cst.n_sample_range),dtype=np.complex64)
        self.tracker_count=np.zeros(n_burst,dtype=np.int64)
        
        # Reading static values from data buffer
        self.burst_power=self.data_buffer.burst_power
        self.burst_phase=self.data_buffer.burst_phase

    def load(self,index_center):

        """
        Loading a new datablock centered on index_center. If the new data block overlaps with the old one, the overlapping data is kept as it is and shifted.
        """

        logging.debug("Loading data block center on index {}.".format(index_center))

        try:
            
            assert(index_center>=0 and index_center<self.data_buffer.n_burst)
            
        except AssertionError:
            
            logging.error("Bad center index.")
        else:

            start_fill=0
            shift = index_center - self.index_center
        
            if self.index_center!=np.nan and shift>0 and shift<self.n_burst:

                self.time[:-shift]=self.time[shift:]
                self.lat[:-shift]=self.lat[shift:]
                self.lon[:-shift]=self.lon[shift:]
                self.surf_type[:-shift]=self.surf_type[shift:]
                self.xs[:-shift]=self.xs[shift:]
                self.ys[:-shift]=self.ys[shift:]
                self.zs[:-shift]=self.zs[shift:]
                self.vxs[:-shift]=self.vxs[shift:]
                self.vys[:-shift]=self.vys[shift:]
                self.vzs[:-shift]=self.vzs[shift:]
                self.alt[:-shift]=self.alt[shift:]
                self.orb_alt_rate[:-shift]=self.orb_alt_rate[shift:]
                self.h0_applied[:-shift]=self.h0_applied[shift:]
                self.cor2_applied[:-shift]=self.cor2_applied[shift:]
                self.tracker[:-shift]=self.tracker[shift:]
                self.cog[:-shift]=self.cog[shift:]
                self.agc[:-shift]=self.agc[shift:]
                self.scale_factor[:-shift]=self.scale_factor[shift:]
                self.sig0_cal[:-shift]=self.sig0_cal[shift:]
                self.i_count[:-shift]=self.i_count[shift:]
                self.q_count[:-shift]=self.q_count[shift:]
                self.burst_count_cycle[:-shift]=self.burst_count_cycle[shift:]
                self.Echo[:-shift]=self.Echo[shift:]
                self.echo[:-shift]=self.echo[shift:]
                self.tracker_count[:-shift]=self.tracker_count[shift:]

                start_fill=self.n_burst-shift

            for k in range(start_fill,self.n_burst):

                ind=index_center + k - self.n_burst//2
                
                if ind>=0 and ind<self.data_buffer.n_burst:

                    self.time[k]=self.data_buffer.time[ind]
                    self.lat[k]=self.data_buffer.lat[ind]
                    self.lon[k]=self.data_buffer.lon[ind]
                    self.surf_type[k]=self.data_buffer.surf_type[ind]
                    self.xs[k]=self.data_buffer.xs[ind]
                    self.ys[k]=self.data_buffer.ys[ind]
                    self.zs[k]=self.data_buffer.zs[ind]
                    self.vxs[k]=self.data_buffer.vxs[ind]
                    self.vys[k]=self.data_buffer.vys[ind]
                    self.vzs[k]=self.data_buffer.vzs[ind]
                    self.alt[k]=self.data_buffer.alt[ind]
                    self.orb_alt_rate[k]=self.data_buffer.orb_alt_rate[ind]
                    self.h0_applied[k]=self.data_buffer.h0_applied[ind]
                    self.cor2_applied[k]=self.data_buffer.cor2_applied[ind]
                    self.tracker[k]=self.data_buffer.tracker[ind]
                    self.cog[k]=self.data_buffer.cog[ind]
                    self.agc[k]=self.data_buffer.agc[ind]
                    self.scale_factor[k]=self.data_buffer.scale_factor[ind]
                    self.sig0_cal[k]=self.data_buffer.sig0_cal[ind]
                    self.i_count[k]=self.data_buffer.i_count[ind]
                    self.q_count[k]=self.data_buffer.q_count[ind]
                    self.burst_count_cycle[k]=self.data_buffer.burst_count_cycle[ind]

                    self.calibrate_burst(k)

                else:

                    self.time[k]=np.nan
                    self.lat[k]=0
                    self.lon[k]=0
                    self.surf_type[k]=0
                    self.xs[k]=0
                    self.ys[k]=0
                    self.zs[k]=0
                    self.vxs[k]=0
                    self.vys[k]=0
                    self.vzs[k]=0
                    self.alt[k]=0
                    self.orb_alt_rate[k]=0
                    self.h0_applied[k]=0
                    self.cor2_applied[k]=0
                    self.tracker[k]=0
                    self.cog[k]=0
                    self.agc[k]=0
                    self.scale_factor[k]=0
                    self.sig0_cal[k]=0
                    self.i_count[k]=0
                    self.q_count[k]=0
                    self.burst_count_cycle[k]=0
                    self.Echo[k]=0
                    self.echo[k]=0
                    self.tracker_count[k]=0

        # Updating the index_center value
        self.index_center=index_center
                
    def calibrate_burst(self,k):

        """
        Calibrate one burst and compute the tracker count and frequency domain pulses.  
        """

        # tracker_count computation
        h=self.h0_applied[k]+(self.burst_count_cycle[k]-1)*((self.cor2_applied[k]/4).astype(np.int)>>4)
        hc=(h/(2**8)).astype(np.int)*(2**8)
        hf=h-hc
        if hf>127:
            hc+=256
            hf+=-256
        self.tracker_count[k]=hc/(2**8)

        # applying calibrations
        echo=self.i_count[k]+1j*self.q_count[k]

        # pi shift between bursts
        if (self.burst_count_cycle[k] % 2) == 1:
            echo*=-1

        # CAL1 phase correction
        echo*=np.exp(1j*self.burst_phase)[:,None]

        # CAL1 power correction
        echo*=np.sqrt(self.burst_power)[:,None]

        # AGC correction
        echo*=np.sqrt(10**(self.agc[k]/10.0))

        # Tracker experimental correction
        echo*=np.exp(1j*self.tracker_count[k]*cst.tracker_phase_shift-2*1j*np.pi*cst.fc*2/cst.c*self.tracker[k])

        # Range FFT
        self.Echo[k]=np.fft.fftshift(np.fft.fft(echo,axis=1),axes=1)

        # RVP and CAL2 correction
        self.Echo[k]*=np.exp(1j*np.pi*cst.alpha*(cst.tau-cst.tau_offset)**2)[None,:] / np.sqrt(self.gprw_mean)[None,:]
        
        # Range IFFT
        self.echo[k]=np.fft.ifft(np.fft.fftshift(self.Echo[k],axes=1),axis=1)
    
