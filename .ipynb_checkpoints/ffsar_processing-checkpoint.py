#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging

import numpy as np
import pyproj
import scipy.interpolate as interpolate

try:
    import numexpr
except ImportError:
    logging.info("Unable to import numexpr. Continuing without it, but processing is slower.")
    use_numexpr=False
else:
    use_numexpr=True
    
from cst import Cst

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

cst=Cst()

class BackprojectionProcessing:
    
    # Suffix : time, latitude and longitude dimensions/variables should be time+suffix, lat+suffix, lon+suffix
    suffix="_ffsar"
    
    def __init__(self,data_block,n_sl_output,n_multilook=None,zp=2,hamming_az=True,hamming_range=True,antenna_gain_compensation=False,range_ext_factor=1):
        
        # =============================================================================
        # Processing parameters                        
        # =============================================================================
        
        # Data block
        self.data_block=data_block
        self.n_burst=self.data_block.n_burst
        
        # Number of single-looks to compute
        self.n_sl_output=n_sl_output
        
        # Range window extension factor
        self.range_ext_factor=range_ext_factor

        # Zeropadding factor in range
        self.zp=zp
        
        # # Hamming window in azimuth
        # self.hamming_az=hamming_az
        # if self.hamming_az:
        #     self.hamming_window_az=0.5-0.5*np.cos(2*np.pi*np.arange(0,cst.n_pulse_burst)/cst.n_pulse_burst)
        #     self.hamming_window_az*=1/np.sum(self.hamming_window_az)*cst.n_pulse_burst
            
        # # Hamming window in range
        # self.hamming_range=hamming_range
        # if self.hamming_range:
        #     self.hamming_window_range=0.5-0.5*np.cos(2*np.pi*np.arange(0,cst.n_sample_range*self.range_ext_factor)/(cst.n_sample_range*self.range_ext_factor))
        #     self.hamming_window_range*=1/np.sum(self.hamming_window_range)*cst.n_sample_range*self.range_ext_factor
        
        # Hamming window in azimuth
        self.hamming_az=hamming_az
        if self.hamming_az:
            self.hamming_window_az=cst.a1-cst.a2*np.cos(2*np.pi*np.arange(0,cst.n_pulse_burst)/cst.n_pulse_burst)
            # self.hamming_window_az*=np.sqrt(cst.n_pulse_burst/np.sum(self.hamming_window_az**2))
            self.hamming_window_az /= np.sqrt(cst.a1**2+0.5*cst.a2**2)

        # Hamming window in range
        self.hamming_range=hamming_range
        self.hamming_window_range=cst.a1-cst.a2*np.cos(2*np.pi*np.arange(0,cst.n_sample_range*self.range_ext_factor)/cst.n_sample_range*self.range_ext_factor)
        # self.hamming_window_range*=np.sqrt(cst.n_sample_range/np.sum(self.hamming_window_range**2))
        self.hamming_window_range /= np.sqrt(cst.a1**2+0.5*cst.a2**2)
        
        # Toggles antenna pattern compensation in azimuth
        self.antenna_gain_compensation=antenna_gain_compensation
        
        # Number of multi-looks to compute
        if n_multilook is not None:
            self.n_multilook=n_multilook
        else:
            self.n_multilook=self.n_sl_output
        self.posting_rate = self.n_multilook/(self.n_sl_output*cst.PRI)
            
        # =============================================================================
        # Processing variables        
        # =============================================================================
            
        # Radial velocity and Doppler centroid
        self.radial_velocity=np.nan
        self.velocity=np.nan
        self.spacing=np.nan
        
        # Doppler frequencies of the bursts
        self.doppler_freqs=np.empty((self.n_multilook,self.n_burst),dtype=np.float)
        
        # RMC coarse
        self.rmc_coarse=np.empty((self.n_multilook,self.n_burst),dtype=np.int32)
        
        # Scale factor
        self.scale_factor=np.nan
        
        # Pulse peakiness
        self.pulse_peakiness=np.empty(self.n_multilook,dtype=np.float)
        
        # Thermal noise estimate
        self.tn_ffsar=np.nan
        self.tn_plrm=np.nan

        # Time and frequency vectors
        self.t=np.arange(-cst.n_sample_range*self.range_ext_factor//2,cst.n_sample_range*self.range_ext_factor//2) / (cst.n_sample_range*self.range_ext_factor)*cst.T       
        self.eta=np.arange(-self.n_sl_output//2,self.n_sl_output//2)*cst.PRI
        
        # Data matrices
        self.data_ext=np.zeros((self.n_burst*cst.n_pulse_burst,cst.n_sample_range*self.range_ext_factor),dtype=np.complex64)       
        self.data=np.zeros((self.n_burst*cst.n_pulse_burst,cst.n_sample_range*self.range_ext_factor),dtype=np.complex64)
        self.H=np.zeros((self.n_burst*cst.n_pulse_burst,cst.n_sample_range*self.range_ext_factor),dtype=np.complex64)

        # Output data
        self.slc=np.zeros((self.n_sl_output,cst.n_sample_range*self.zp),dtype=np.complex64)
        self.multilook=np.zeros((self.n_multilook,cst.n_sample_range*self.zp),dtype=np.float)
        self.msc=np.zeros((self.n_multilook,cst.n_sample_range*self.zp),dtype=np.float)      
        self.time_multilook=np.zeros(self.n_multilook,dtype=np.float)
        self.lat_multilook=np.zeros(self.n_multilook,dtype=np.float)
        self.lon_multilook=np.zeros(self.n_multilook,dtype=np.float)
        self.tracker_multilook=np.zeros(self.n_multilook,dtype=np.float)
        self.alt_multilook=np.zeros(self.n_multilook,dtype=np.float)

        # PLRM : the PRLM is part of the code here for the estimation of the thermal noise level 1b output variable used in level 2 retrackers
        self.h_plrm=np.empty((4,cst.n_pulse_burst,cst.n_sample_range),dtype=np.complex)
        self.echo_foc_plrm=np.empty((4,cst.n_pulse_burst,cst.n_sample_range),dtype=np.complex)
        self.stack_complex_plrm=np.empty((4,cst.n_pulse_burst,cst.n_sample_range*self.zp),dtype=np.complex)
        self.stack_plrm=np.empty((4,cst.n_pulse_burst,cst.n_sample_range*self.zp),dtype=np.float)
        self.multilook_plrm=np.empty(cst.n_sample_range*self.zp,dtype=np.float)
        self.rmc_plrm=np.empty((4,cst.n_pulse_burst),dtype=np.float)

    def get_info(self):
        """
        
        :return: DESCRIPTION
        :rtype: TYPE

        """
        
        dict_dimensions={}
        dict_dimensions['time_ffsar']=None
        dict_dimensions['echo_sample_ffsar']=cst.n_sample_range*self.zp
        dict_dimensions['stack_sample_ffsar']=self.n_burst
        
        dict_variables={}
        dict_variables['time_ffsar']={'type':np.float64,
                                            'dimension':['time_ffsar'],
                                            'attributes':{'units':'seconds since 2000-01-01 00:00:00.0',
                                                          'long_name':'UTC time FFSAR'},
                                            }
        dict_variables['lat_ffsar']={'type':np.float64,
                                           'dimension':['time_ffsar'],
                                           'attributes':{'units':'degrees_north',
                                                         'long_name':'Latitude'}
                                           }
        dict_variables['lon_ffsar']={'type':np.float64,
                                           'dimension':['time_ffsar'],
                                           'attributes':{'units':'degrees_east',
                                                         'long_name':'Longitude'}
                                           }
        dict_variables['alt_ffsar']={'type':np.float64,
                                           'dimension':['time_ffsar'],
                                           'attributes':{'units':'m',
                                                         'long_name':'Platform altitude'},
                                           'coordinates':['lon_ffsar','lat_ffsar']
                                           }
        dict_variables['tracker_ffsar']={'type':np.float64,
                                           'dimension':['time_ffsar'],
                                           'attributes':{'units':'m',
                                                         'long_name':'Tracker range from the reference gate (=43 for S3 starting from zero index)'},
                                           'coordinates':['lon_ffsar','lat_ffsar']
                                           }
        dict_variables['radial_velocity_ffsar']={'type':np.float64,
                                           'dimension':['time_ffsar'],
                                           'attributes':{'units':'m/s',
                                                         'long_name':'Platform radial velocity'},
                                           'coordinates':['lon_ffsar','lat_ffsar']
                                           }
        dict_variables['velocity_ffsar']={'type':np.float64,
                                           'dimension':['time_ffsar'],
                                           'attributes':{'units':'m/s',
                                                         'long_name':'Platform velocity'},
                                           'coordinates':['lon_ffsar','lat_ffsar']
                                           }
        dict_variables['separation_between_waveform_ffsar']={'type':np.float64,
                                           'dimension':['time_ffsar'],
                                           'attributes':{'units':'m',
                                                         'long_name':'Separation between of multilooked waveforms on ground.'},
                                           'coordinates':['lon_ffsar','lat_ffsar']
                                           }
        dict_variables['multilook_ffsar']={'type':np.uint64,
                                           'dimension':['time_ffsar','echo_sample_ffsar'],
                                           'attributes':{'units':'count',
                                                         'long_name':'Multilooked waveform; FFSAR multi-looking is processed by averaging coherent single-look power waveforms',
                                                         'scale_factor':0.001,
                                                         'add_offset':0.0}
                                           }
        dict_variables['msc_ffsar']={'type':np.uint16,
                                           'dimension':['time_ffsar','echo_sample_ffsar'],
                                           'attributes':{'units':'None',
                                                         'long_name':'Magnitude Squared Coherence (MSC) of single-looks; MSC is equal to the ratio of the power of the average of single-look and the average of the power of singe-look, the values are between 0 (weak coherence) and 1 (strong coherence)',
                                                         'scale_factor':1/(2**16),
                                                         'add_offset':0.0}
                                           }
        dict_variables['scale_factor_ffsar']={'type':np.float64,
                                           'dimension':['time_ffsar'],
                                           'attributes':{'units':'dB',
                                                         'long_name':'Scale factor for sig0 evaluation'},
                                           'coordinates':['lon_ffsar','lat_ffsar']
                                           }
        dict_variables['pulse_peakiness_ffsar']={'type':np.float64,
                                           'dimension':['time_ffsar'],
                                           'attributes':{'units':'None',
                                                         'long_name':'Pulse peakiness is equal to the ratio between the maximum and the mean values of the FFSAR multi-looked waveform multiplied by 85/128'},
                                           'coordinates':['lon_ffsar','lat_ffsar']        
                                           }
        dict_variables['doppler_freqs_ffsar']={'type':np.float64,
                                               'dimension':['time_ffsar','stack_sample_ffsar'],
                                               'attributes':{'units':'Hz',
                                                             'long_name':'Observation Doppler frequencies'},
                                               }
        dict_variables['rmc_coarse_ffsar']={'type':np.int32,
                                            'dimension':['time_ffsar','stack_sample_ffsar'],
                                            'attributes':{'units':'bin count',
                                                          'long_name':'Coarse part of the range migration correction for mask computation. When applying RMC, looks are shifted by rmc_coarse*zp oversampled range bins to the right.'},
                                            }
        dict_variables['tn_ffsar']={'type':np.float64,
                                    'dimension':['time_ffsar'],
                                    'attributes':{'units':'count',
                                                  'long_name':'Estimate of the thermal noise level on the FFSAR multilook; Thermal noise is calculated by averaging from the 15th (multiplied the zero-padding) to the 20th (multiplied the zero-padding) range gates of PLRM multilook'},
                                    'coordinates':['lon_ffsar','lat_ffsar']
                                    }
        return dict_dimensions,dict_variables,self.suffix
        
    def get_data(self):
        """
        

        Returns
        -------
        None.

        """
        
        dict_data={}
        dict_data['time_ffsar']=self.time_multilook
        dict_data['lat_ffsar']=self.lat_multilook
        dict_data['lon_ffsar']=self.lon_multilook
        dict_data['alt_ffsar']=self.alt_multilook
        dict_data['tracker_ffsar']=self.tracker_multilook
        dict_data['radial_velocity_ffsar']=self.radial_velocity
        dict_data['velocity_ffsar']=self.velocity
        dict_data['separation_between_waveform_ffsar'] = self.spacing
        dict_data['multilook_ffsar']=self.multilook
        dict_data['msc_ffsar']=self.msc
        dict_data['scale_factor_ffsar']=self.scale_factor
        dict_data['doppler_freqs_ffsar']=self.doppler_freqs
        dict_data['rmc_coarse_ffsar']=self.rmc_coarse
        dict_data['tn_ffsar']=self.tn_ffsar
        dict_data['pulse_peakiness_ffsar']=self.pulse_peakiness
        
        return dict_data

    def process(self,delay_range=0,ind_burst_process=None):
        
        # Selecting bursts to process
        if ind_burst_process is None:
            list_ind_burst=list(np.arange(self.n_burst))
        else:
            list_ind_burst=ind_burst_process
        n_burst_process=len(list_ind_burst)
        
        # =============================================================================
        # Computing radial velocity
        # =============================================================================
        
        lon_focus,lat_focus,alt_focus=pyproj.transform(cst.ecef,cst.lla,
                                                       self.data_block.xs[self.n_burst//2],
                                                       self.data_block.ys[self.n_burst//2], 
                                                       self.data_block.zs[self.n_burst//2])
        
        # Setting the focusing point at elevation alt-tracker (estimation of the real elevation)
        x_focus,y_focus,z_focus=pyproj.transform(cst.lla,cst.ecef,
                                                 lon_focus,
                                                 lat_focus,
                                                 alt_focus-self.data_block.tracker[self.n_burst//2])
        
        elev_focus=self.data_block.alt[self.n_burst//2]-self.data_block.tracker[self.n_burst//2]
        
        self.radial_velocity=-(self.data_block.vxs[self.n_burst//2]*(x_focus-self.data_block.xs[self.n_burst//2])
                               + self.data_block.vys[self.n_burst//2]*(y_focus-self.data_block.ys[self.n_burst//2])
                               + self.data_block.vzs[self.n_burst//2]*(z_focus-self.data_block.zs[self.n_burst//2]))/self.data_block.tracker[self.n_burst//2]

        self.velocity=np.sqrt(self.data_block.vxs[self.n_burst//2]**2
                              +self.data_block.vys[self.n_burst//2]**2
                              +self.data_block.vzs[self.n_burst//2]**2)
        self.spacing=1.2*self.velocity/self.posting_rate
        
        rs=np.sqrt((self.data_block.xs-x_focus)**2 + (self.data_block.ys-y_focus)**2 + (self.data_block.zs-z_focus)**2)
        
        fds=-2/cst.c*cst.fc*(self.data_block.vxs*(x_focus-self.data_block.xs)
                             + self.data_block.vys*(y_focus-self.data_block.ys)
                             + self.data_block.vzs*(z_focus-self.data_block.zs))/rs
        
        self.doppler_freqs[:]=(fds - 2*cst.fc/cst.c*self.data_block.orb_alt_rate)[None,:]

        self.rmc_coarse[:]=np.round((rs-rs[self.n_burst//2]-self.data_block.tracker+self.data_block.tracker[self.n_burst//2] - fds*cst.c/2/cst.alpha)/cst.c*2*cst.B).astype(np.int)[None,:]
        
        # =============================================================================
        # Interpolation of the position and the velocity        
        # =============================================================================
        
        # Time tag of the pulses
        eta_pulse=self.data_block.time[:,None]-self.data_block.time[self.n_burst//2]+cst.eta_burst[None,:]
        # eta_pulse=np.arange(-self.n_burst//2,self.n_burst//2)[:,None]*cst.BRI + cst.eta_burst[None,:]
        
        # Time tag of the bursts
        eta_burst=eta_pulse[:,cst.n_pulse_burst//2]
        
        # Interpolation functions
        fx=np.poly1d(np.polyfit(eta_burst,self.data_block.xs,4))
        fy=np.poly1d(np.polyfit(eta_burst,self.data_block.ys,4))
        fz=np.poly1d(np.polyfit(eta_burst,self.data_block.zs,4))
        fvx=np.poly1d(np.polyfit(eta_burst,self.data_block.vxs,4))
        fvy=np.poly1d(np.polyfit(eta_burst,self.data_block.vys,4))
        fvz=np.poly1d(np.polyfit(eta_burst,self.data_block.vzs,4))
        
        x_pulse,y_pulse,z_pulse=fx(eta_pulse[list_ind_burst].ravel()),fy(eta_pulse[list_ind_burst].ravel()),fz(eta_pulse[list_ind_burst].ravel())
        vx_pulse,vy_pulse,vz_pulse=fvx(eta_pulse[list_ind_burst].ravel()),fvy(eta_pulse[list_ind_burst].ravel()),fvz(eta_pulse[list_ind_burst].ravel())
        tracker_pulse=np.repeat(self.data_block.tracker[list_ind_burst],cst.n_pulse_burst)
        
        # =============================================================================
        # Extending the window and range IFFT        
        # =============================================================================
        
        if self.hamming_az:
            self.data_ext[:n_burst_process*cst.n_pulse_burst,cst.n_sample_range*self.range_ext_factor//2-cst.n_sample_range//2:
                          cst.n_sample_range*self.range_ext_factor//2+cst.n_sample_range//2]=np.reshape(self.data_block.Echo[list_ind_burst]*self.hamming_window_az[None,:,None], (n_burst_process*cst.n_pulse_burst,cst.n_sample_range))
        else:
            self.data_ext[:n_burst_process*cst.n_pulse_burst,cst.n_sample_range*self.range_ext_factor//2-cst.n_sample_range//2:
                          cst.n_sample_range*self.range_ext_factor//2+cst.n_sample_range//2]=np.reshape(self.data_block.Echo[list_ind_burst], (n_burst_process*cst.n_pulse_burst,cst.n_sample_range))
            
        self.data[:n_burst_process*cst.n_pulse_burst]=np.fft.ifft(np.fft.fftshift(self.data_ext[:n_burst_process*cst.n_pulse_burst,:],axes=1),axis=1)
        
        if self.hamming_range:
            self.data*=self.hamming_window_range[None,:]
            
        # =============================================================================
        # Focusing points        
        # =============================================================================
            
        lon_focus,lat_focus,alt_focus=pyproj.transform(cst.ecef, cst.lla,fx(self.eta),fy(self.eta),fz(self.eta),radians=False)
        xn_focus,yn_focus,zn_focus=pyproj.transform(cst.lla, cst.ecef, lon_focus, lat_focus, np.ones(self.n_sl_output)*elev_focus)
        
        # =============================================================================
        # Loop over focusing points        
        # =============================================================================
        
        diff_eta=np.repeat(self.eta[self.n_sl_output//self.n_multilook//2::self.n_sl_output//self.n_multilook],self.n_sl_output//self.n_multilook)
        
        for j in range(self.n_sl_output):
            
            r_p=np.sqrt((x_pulse-xn_focus[j])**2 + (y_pulse-yn_focus[j])**2 + (z_pulse-zn_focus[j])**2)
            fd_p=-2/cst.c*cst.fc*(vx_pulse * (xn_focus[j] - x_pulse)
                                  +vy_pulse * (yn_focus[j] - y_pulse)
                                  +vz_pulse * (zn_focus[j] - z_pulse)) / r_p

            corr_doppler=-fd_p*cst.c/cst.alpha/2

            corr_range=r_p - alt_focus[j] + elev_focus + corr_doppler - tracker_pulse + self.data_block.tracker[self.n_burst//2] - delay_range - (alt_focus[self.n_sl_output//2]-alt_focus[j]) - diff_eta[j]*self.radial_velocity

            t_comp_h=self.t[None,:]
            corr_range_comp_h=corr_range[:,None]
            r_p_comp_h=r_p[:,None]
            fc,alpha,c,gamma,pi=cst.fc,cst.alpha,cst.c,cst.gamma,np.pi
            if use_numexpr:
                self.H[:n_burst_process*cst.n_pulse_burst]=numexpr.evaluate("exp(-2*1j*pi*alpha*t_comp_h*2/c*corr_range_comp_h + 1j*2*pi*2/c*fc*r_p_comp_h)")
            else:
                self.H[:n_burst_process*cst.n_pulse_burst]=np.exp(-2*1j*pi*alpha*t_comp_h*2/c*corr_range_comp_h + 1j*2*pi*2/c*fc*r_p_comp_h)

            self.slc[j,:] = np.fft.fftshift(
                np.fft.fft(
                    np.sum(self.data[:n_burst_process*cst.n_pulse_burst,:]*self.H[:n_burst_process*cst.n_pulse_burst,:],axis=0),n=cst.n_sample_range*self.range_ext_factor*self.zp))[cst.n_sample_range*self.range_ext_factor*self.zp//2-cst.n_sample_range*self.zp//2:cst.n_sample_range*self.range_ext_factor*self.zp//2+cst.n_sample_range*self.zp//2]
        
        # Scaling
        self.slc[:]*=10**(-self.data_block.agc[self.n_burst//2]/20) / np.sqrt(cst.n_pulse_burst) / (cst.n_sample_range)
        
        # Multilooking
        self.multilook[:]=np.mean(np.reshape(np.real(self.slc*np.conjugate(self.slc)),(self.n_multilook,-1,cst.n_sample_range*self.zp)),axis=1)
        self.msc[:]=np.abs(np.mean(np.reshape(self.slc,(self.n_multilook,-1,cst.n_sample_range*self.zp)),axis=1))**2 / self.multilook
        
        # Time tag of multilooks
        self.time_multilook[:]=self.data_block.time[self.n_burst//2]+self.eta[self.n_sl_output//self.n_multilook//2::self.n_sl_output//self.n_multilook]
        
        # Interplation of the latitude and longitude of the multilooks
        f_lat=interpolate.interp1d(eta_burst,self.data_block.lat,kind="cubic")
        f_lon=interpolate.interp1d(eta_burst,np.unwrap(self.data_block.lon,discont=180),kind="cubic")
        self.lon_multilook[:]=f_lon(self.time_multilook-self.data_block.time[self.n_burst//2])
        self.lat_multilook[:]=f_lat(self.time_multilook-self.data_block.time[self.n_burst//2])
        
        # Adding the COG to the tracker
        self.tracker_multilook[:]=self.data_block.tracker[self.n_burst//2] + self.data_block.cog[self.n_burst//2]
        # self.alt_multilook[:]=self.data_block.alt[self.n_burst//2]
        self.alt_multilook[:]=self.data_block.alt[self.n_burst//2] + self.radial_velocity*self.eta[self.n_sl_output//self.n_multilook//2::self.n_sl_output//self.n_multilook]
        
        # Scale factor
        # self.scale_factor=self.data_block.scale_factor[self.n_burst//2]
        self.scale_factor=self.data_block.agc[self.n_burst//2] + self.data_block.sig0_cal[self.n_burst//2] + 30*np.log10(self.data_block.alt[self.n_burst//2]) + 10*np.log10(1+self.data_block.alt[self.n_burst//2]/cst.geoid_wgs.a) + 10*np.log10(4*16*np.pi**2*cst.fc**2/cst.c**3/256/cst.Fs) -2*cst.GdB
        
        # Pulse peakiness
        self.pulse_peakiness[:]=np.max(self.multilook,axis=1)/np.mean(self.multilook,axis=1) * 85/128
        
        # =============================================================================
        # PLRM waveform computation and thermal noise estimation
        # =============================================================================
        
        # Range migration correction : radial velocity + Doppler correction
        self.rmc_plrm[:]=self.radial_velocity*(np.arange(-2,2)[:,None]*cst.BRI+cst.eta_burst[None,:]) - self.data_block.tracker[self.n_burst//2-2:self.n_burst//2+2:,None] + self.data_block.tracker[self.n_burst//2] - 2/cst.c*cst.fc*self.radial_velocity*cst.c/2/cst.alpha 
        
        # Computation of the focusing operator
        self.h_plrm[:]=np.exp(-2*1j*np.pi*cst.alpha*cst.t[None,None,:]*2/cst.c*self.rmc_plrm[:,:,None])
        
        # Applying range migrations
        self.echo_foc_plrm[:]=self.data_block.echo[self.n_burst//2-2:self.n_burst//2+2,:,:]*self.h_plrm

        # Thermal noise estimation : the PLRM echo is computed with a Hamming window
        self.stack_complex_plrm[:]=np.fft.fftshift(np.fft.fft(self.echo_foc_plrm*self.hamming_window_range[None,None,::self.range_ext_factor],axis=2,n=cst.n_sample_range*self.zp),axes=2)
        self.stack_complex_plrm*=10**(-self.data_block.agc[self.n_burst//2]/20) / cst.n_sample_range
        self.stack_plrm[:]=self.stack_complex_plrm.real**2 + self.stack_complex_plrm.imag**2
        np.sum(self.stack_plrm,axis=(0,1),out=self.multilook_plrm)
        self.tn_plrm=np.mean(self.multilook_plrm[15*self.zp:20*self.zp])
        self.tn_ffsar=self.n_burst/256*self.tn_plrm
        

class OmegakProcessing:
    
    # Suffix : time, latitude and longitude dimensions/variables should be time+suffix, lat+suffix, lon+suffix
    suffix="_ffsar"
    
    def __init__(self,data_block,n_fft_az,n_sl_output,n_multilook=None,zp=2,hamming_az=False,hamming_range=False,antenna_gain_compensation=False,range_ext_factor=1):
        
        # =============================================================================
        # Processing parameters                        
        # =============================================================================
        
        # Data block
        self.data_block=data_block
        self.n_burst=self.data_block.n_burst
        
        # Size of the matrix for the azimuth FFT
        self.n_fft_az=n_fft_az
        
        # Number of single-looks to compute
        self.n_sl_output=n_sl_output
        
        # Range window extension factor
        self.range_ext_factor=range_ext_factor

        # Zeropadding factor in range
        self.zp=zp
        
        # # Hamming window in azimuth
        # self.hamming_az=hamming_az
        # if self.hamming_az:
        #     self.hamming_window_az=0.5-0.5*np.cos(2*np.pi*np.arange(0,cst.n_pulse_burst)/cst.n_pulse_burst)
        #     # self.hamming_window_az*=1/np.sum(self.hamming_window_az)*cst.n_pulse_burst
        #     self.hamming_window_az*=np.sqrt(cst.n_pulse_burst/np.sum(self.hamming_window_az**2))

        # # Hamming window in range
        # self.hamming_range=hamming_range
        # self.hamming_window_range=0.5-0.5*np.cos(2*np.pi*np.arange(0,cst.n_sample_range*self.range_ext_factor)/(cst.n_sample_range*self.range_ext_factor))
        # # self.hamming_window_range*=1/np.sum(self.hamming_window_range)*cst.n_sample_range*self.range_ext_factor
        # self.hamming_window_range*=np.sqrt(cst.n_sample_range/np.sum(self.hamming_window_range**2))

        # Hamming window in azimuth
        self.hamming_az=hamming_az
        if self.hamming_az:
            self.hamming_window_az=cst.a1-cst.a2*np.cos(2*np.pi*np.arange(0,cst.n_pulse_burst)/cst.n_pulse_burst)
            # self.hamming_window_az*=np.sqrt(cst.n_pulse_burst/np.sum(self.hamming_window_az**2))
            self.hamming_window_az /= np.sqrt(cst.a1**2+0.5*cst.a2**2)

        # Hamming window in range
        self.hamming_range=hamming_range
        self.hamming_window_range=cst.a1-cst.a2*np.cos(2*np.pi*np.arange(0,cst.n_sample_range*self.range_ext_factor)/cst.n_sample_range*self.range_ext_factor)
        # self.hamming_window_range*=np.sqrt(cst.n_sample_range/np.sum(self.hamming_window_range**2))
        self.hamming_window_range /= np.sqrt(cst.a1**2+0.5*cst.a2**2)

        # Toggles antenna pattern compensation in azimuth
        self.antenna_gain_compensation=antenna_gain_compensation
        
        # Number of multi-looks to compute
        if n_multilook is not None:
            self.n_multilook=n_multilook
        else:
            self.n_multilook=self.n_sl_output
        self.posting_rate = self.n_multilook/(self.n_sl_output*cst.PRI)
            
        # =============================================================================
        # Processing variables        
        # =============================================================================
            
        # Radial velocity and Doppler centroid
        self.radial_velocity=np.nan
        self.velocity=np.nan
        self.spacing=np.nan
        self.doppler_centroid=np.nan
        
        # Doppler frequencies of the bursts
        self.doppler_freqs=np.empty((self.n_multilook,self.n_burst),dtype=np.float)
        
        # RMC coarse
        self.rmc_coarse=np.empty((self.n_multilook,self.n_burst),dtype=np.int32)
        
        # Scale factor
        self.scale_factor=np.nan
        
        # Pulse peakiness
        self.pulse_peakiness=np.empty(self.n_multilook,dtype=np.float)
        
        # Thermal noise estimate
        self.tn_ffsar=np.nan
        self.tn_plrm=np.nan

        # Time and frequency vectors
        self.t=np.arange(-cst.n_sample_range*self.range_ext_factor//2,cst.n_sample_range*self.range_ext_factor//2) / (cst.n_sample_range*self.range_ext_factor)*cst.T       
        self.eta=np.arange(-self.n_fft_az//2,self.n_fft_az//2)*cst.PRI
        self.fa=np.fft.fftfreq(self.n_fft_az,d=cst.PRI)
        self.fa_shift=np.fft.fftshift(self.fa)
        
        # Time and frequency for stationnary phase point (SPP) computation
        self.fa_spp=np.hstack((self.fa_shift[::256],self.fa_shift[-1]))
        self.t_spp=np.hstack((cst.t[::4],cst.t[-1]))
        self.coeffs_poly_spp=np.zeros((self.fa_spp.size,self.t_spp.size,4),dtype=np.complex64)
        
        # Data matrices
        self.data_burst_ext=np.zeros((cst.n_pulse_burst,cst.n_sample_range*self.range_ext_factor),dtype=np.complex64)       
        self.data=np.zeros((self.n_fft_az,cst.n_sample_range*self.range_ext_factor),dtype=np.complex64)
        self.Data=np.zeros((self.n_fft_az,cst.n_sample_range*self.range_ext_factor),dtype=np.complex64)
        self.spp=np.zeros((self.n_fft_az,cst.n_sample_range*self.range_ext_factor),dtype=np.float)
        self.H=np.zeros((self.n_fft_az,cst.n_sample_range*self.range_ext_factor),dtype=np.complex64)

        # Output data
        self.slc=np.zeros((self.n_sl_output,cst.n_sample_range*self.zp),dtype=np.complex64)
        self.multilook=np.zeros((self.n_multilook,cst.n_sample_range*self.zp),dtype=np.float)
        self.msc=np.zeros((self.n_multilook,cst.n_sample_range*self.zp),dtype=np.float)      
        self.time_multilook=np.zeros(self.n_multilook,dtype=np.float)
        self.lat_multilook=np.zeros(self.n_multilook,dtype=np.float)
        self.lon_multilook=np.zeros(self.n_multilook,dtype=np.float)
        self.tracker_multilook=np.zeros(self.n_multilook,dtype=np.float)
        self.alt_multilook=np.zeros(self.n_multilook,dtype=np.float)
        
        # PLRM : the PRLM is part of the code here for the estimation of the thermal noise level 1b output variable used in level 2 retrackers
        self.h_plrm=np.empty((4,cst.n_pulse_burst,cst.n_sample_range),dtype=np.complex)
        self.echo_foc_plrm=np.empty((4,cst.n_pulse_burst,cst.n_sample_range),dtype=np.complex)
        self.stack_complex_plrm=np.empty((4,cst.n_pulse_burst,cst.n_sample_range*self.zp),dtype=np.complex)
        self.stack_plrm=np.empty((4,cst.n_pulse_burst,cst.n_sample_range*self.zp),dtype=np.float)
        self.multilook_plrm=np.empty(cst.n_sample_range*self.zp,dtype=np.float)
        self.rmc_plrm=np.empty((4,cst.n_pulse_burst),dtype=np.float)
        
    def get_info(self):
        """
        
        :return: DESCRIPTION
        :rtype: TYPE

        """
        
        dict_dimensions={}
        dict_dimensions['time_ffsar']=None
        dict_dimensions['echo_sample_ffsar']=cst.n_sample_range*self.zp
        dict_dimensions['stack_sample_ffsar']=self.n_burst
        
        dict_variables={}
        dict_variables['time_ffsar']={'type':np.float64,
                                            'dimension':['time_ffsar'],
                                            'attributes':{'units':'seconds since 2000-01-01 00:00:00.0',
                                                          'long_name':'UTC time FFSAR'},
                                            }
        dict_variables['lat_ffsar']={'type':np.float64,
                                           'dimension':['time_ffsar'],
                                           'attributes':{'units':'degrees_north',
                                                         'long_name':'Latitude'}
                                           }
        dict_variables['lon_ffsar']={'type':np.float64,
                                           'dimension':['time_ffsar'],
                                           'attributes':{'units':'degrees_east',
                                                         'long_name':'Longitude'}
                                           }
        dict_variables['alt_ffsar']={'type':np.float64,
                                           'dimension':['time_ffsar'],
                                           'attributes':{'units':'m',
                                                         'long_name':'Platform altitude'},
                                           'coordinates':['lon_ffsar','lat_ffsar']
                                           }
        dict_variables['tracker_ffsar']={'type':np.float64,
                                           'dimension':['time_ffsar'],
                                           'attributes':{'units':'m',
                                                         'long_name':'Tracker range from the reference gate (=43 for S3 starting from zero index)'},
                                           'coordinates':['lon_ffsar','lat_ffsar']
                                           }
        dict_variables['radial_velocity_ffsar']={'type':np.float64,
                                           'dimension':['time_ffsar'],
                                           'attributes':{'units':'m/s',
                                                         'long_name':'Platform radial velocity'},
                                           'coordinates':['lon_ffsar','lat_ffsar']
                                           }
        dict_variables['velocity_ffsar']={'type':np.float64,
                                           'dimension':['time_ffsar'],
                                           'attributes':{'units':'m/s',
                                                         'long_name':'Platform velocity'},
                                           'coordinates':['lon_ffsar','lat_ffsar']
                                           }
        dict_variables['separation_between_waveform_ffsar']={'type':np.float64,
                                           'dimension':['time_ffsar'],
                                           'attributes':{'units':'m',
                                                         'long_name':'Separation between of multilooked waveforms on ground.'},
                                           'coordinates':['lon_ffsar','lat_ffsar']
                                           }
        dict_variables['multilook_ffsar']={'type':np.uint64,
                                           'dimension':['time_ffsar','echo_sample_ffsar'],
                                           'attributes':{'units':'count',
                                                         'long_name':'Multilooked waveform; FFSAR multi-looking is processed by averaging coherent single-look power waveforms',
                                                         'scale_factor':0.001,
                                                         'add_offset':0.0}
                                           }
        dict_variables['msc_ffsar']={'type':np.uint16,
                                           'dimension':['time_ffsar','echo_sample_ffsar'],
                                           'attributes':{'units':'None',
                                                         'long_name':'Magnitude Squared Coherence (MSC) of single-looks; MSC is equal to the ratio of the power of the average of single-look and the average of the power of singe-look, the values are between 0 (weak coherence) and 1 (strong coherence)',
                                                         'scale_factor':1/(2**16),
                                                         'add_offset':0.0}
                                           }
        dict_variables['scale_factor_ffsar']={'type':np.float64,
                                           'dimension':['time_ffsar'],
                                           'attributes':{'units':'dB',
                                                         'long_name':'Scale factor for sig0 evaluation'},
                                           'coordinates':['lon_ffsar','lat_ffsar']        
                                           }
        dict_variables['pulse_peakiness_ffsar']={'type':np.float64,
                                           'dimension':['time_ffsar'],
                                           'attributes':{'units':'None',
                                                         'long_name':'Pulse peakiness is equal to the ratio between the maximum and the mean values of the FFSAR multi-looked waveform multiplied by 85/128'},
                                           'coordinates':['lon_ffsar','lat_ffsar']        
                                           }
        dict_variables['doppler_freqs_ffsar']={'type':np.float64,
                                               'dimension':['time_ffsar','stack_sample_ffsar'],
                                               'attributes':{'units':'Hz',
                                                             'long_name':'Observation Doppler frequencies'},
                                               }
        dict_variables['rmc_coarse_ffsar']={'type':np.int32,
                                            'dimension':['time_ffsar','stack_sample_ffsar'],
                                            'attributes':{'units':'bin count',
                                                          'long_name':'Coarse part of the range migration correction for mask computation. When applying RMC, looks are shifted by rmc_coarse*zp oversampled range bins to the right.'},
                                            }
        dict_variables['tn_ffsar']={'type':np.float64,
                                    'dimension':['time_ffsar'],
                                    'attributes':{'units':'count',
                                                  'long_name':'Estimate of the thermal noise level on the FFSAR multilook; Thermal noise is calculated by averaging from the 15th (multiplied the zero-padding) to the 20th (multiplied the zero-padding) range gates of PLRM multilook'},
                                    'coordinates':['lon_ffsar','lat_ffsar']
                                    }
        return dict_dimensions,dict_variables,self.suffix
        
    def get_data(self):
        """
        

        Returns
        -------
        None.

        """
        
        dict_data={}
        dict_data['time_ffsar']=self.time_multilook
        dict_data['lat_ffsar']=self.lat_multilook
        dict_data['lon_ffsar']=self.lon_multilook
        dict_data['alt_ffsar']=self.alt_multilook
        dict_data['tracker_ffsar']=self.tracker_multilook
        dict_data['radial_velocity_ffsar']=self.radial_velocity
        dict_data['velocity_ffsar']=self.velocity
        dict_data['separation_between_waveform_ffsar'] = self.spacing
        dict_data['multilook_ffsar']=self.multilook
        dict_data['msc_ffsar']=self.msc
        dict_data['scale_factor_ffsar']=self.scale_factor
        dict_data['doppler_freqs_ffsar']=self.doppler_freqs
        dict_data['rmc_coarse_ffsar']=self.rmc_coarse
        dict_data['tn_ffsar']=self.tn_ffsar
        dict_data['pulse_peakiness_ffsar']=self.pulse_peakiness
        
        return dict_data

    def process(self,update_h=True,delay_az=0,delay_range=0,ind_burst_process=None):

        # =============================================================================
        # Computing radial velocity
        # =============================================================================
        
        lon_focus,lat_focus,alt_focus=pyproj.transform(cst.ecef,cst.lla,
                                                       self.data_block.xs[self.n_burst//2],
                                                       self.data_block.ys[self.n_burst//2], 
                                                       self.data_block.zs[self.n_burst//2])
        
        # Setting the focusing point at elevation alt-tracker (estimation of the real elevation)
        x_focus,y_focus,z_focus=pyproj.transform(cst.lla,cst.ecef,
                                                 lon_focus,
                                                 lat_focus,
                                                 alt_focus-self.data_block.tracker[self.n_burst//2])
        
        self.radial_velocity=-(self.data_block.vxs[self.n_burst//2]*(x_focus-self.data_block.xs[self.n_burst//2])
                               + self.data_block.vys[self.n_burst//2]*(y_focus-self.data_block.ys[self.n_burst//2])
                               + self.data_block.vzs[self.n_burst//2]*(z_focus-self.data_block.zs[self.n_burst//2]))/self.data_block.tracker[self.n_burst//2]

        self.doppler_centroid=2/cst.c*cst.fc*self.radial_velocity

        self.velocity=np.sqrt(self.data_block.vxs[self.n_burst//2]**2
                              +self.data_block.vys[self.n_burst//2]**2
                              +self.data_block.vzs[self.n_burst//2]**2)
        self.spacing = 1.2 * self.velocity / self.posting_rate
        
        rs=np.sqrt((self.data_block.xs-x_focus)**2 + (self.data_block.ys-y_focus)**2 + (self.data_block.zs-z_focus)**2)
        
        fds=-2/cst.c*cst.fc*(self.data_block.vxs*(x_focus-self.data_block.xs)
                             + self.data_block.vys*(y_focus-self.data_block.ys)
                             + self.data_block.vzs*(z_focus-self.data_block.zs))/rs
        
        self.doppler_freqs[:]=(fds - 2*cst.fc/cst.c*self.data_block.orb_alt_rate)[None,:]

        self.rmc_coarse[:]=np.round((rs-rs[self.n_burst//2]-self.data_block.tracker+self.data_block.tracker[self.n_burst//2] - fds*cst.c/2/cst.alpha)/cst.c*2*cst.B).astype(np.int)[None,:]
        
        # =============================================================================
        # Building data block : data from the bursts is aligned in range taking into account
        # tracker variations and radial velocity at the center of the block. The RMC is 
        # decomposed into coarse and fine parts. The Doppler centroid correction is also aplied.
        # =============================================================================
        
        # Time tag of the pulses
        # eta_pulse=self.data_block.time[:,None]-self.data_block.time[self.n_burst//2]+cst.eta_burst[None,:]
        eta_pulse=np.arange(-self.n_burst//2,self.n_burst//2)[:,None]*cst.BRI + cst.eta_burst[None,:]
        
        # Time tag of the bursts
        eta_burst=eta_pulse[:,cst.n_pulse_burst//2]
        
        # Range migration correction
        rmc=self.radial_velocity*eta_burst - self.data_block.tracker + self.data_block.tracker[self.n_burst//2] # in m
        rmc_coarse=np.round(rmc/cst.c*2*cst.B).astype(np.int) # in gate
        rmc_fine=rmc-rmc_coarse*cst.c/2/cst.B  # in meter
        
        # Setting to zero the new data block
        self.data[:]=0.0
        
        # Selecting bursts to process
        if ind_burst_process is None:
            list_ind_burst=np.arange(self.n_burst)
        else:
            list_ind_burst=ind_burst_process
            
        # Loop over the bursts of the block
        for b in list_ind_burst:

            # If range migration leave some data inside the window
            if rmc_coarse[b]<cst.n_sample_range*self.range_ext_factor and rmc_coarse[b]>-cst.n_sample_range*self.range_ext_factor:
                
                self.data_burst_ext[:]=0.0

                self.data_burst_ext[:,cst.n_sample_range*self.range_ext_factor//2-cst.n_sample_range//2:
                                    cst.n_sample_range*self.range_ext_factor//2+cst.n_sample_range//2]=self.data_block.Echo[b,:,:]

                # Rolling the data
                self.data_burst_ext[:]=np.roll(self.data_burst_ext,-rmc_coarse[b],axis=1)
                
                # Setting to zero samples affected by RMC
                if rmc_coarse[b]<0:
                    self.data_burst_ext[:,:-rmc_coarse[b]]=0.0
                if rmc_coarse[b]>0:
                    self.data_burst_ext[:,-rmc_coarse[b]:]=0.0

                # Phase correction due to rolling the data
                if rmc_coarse[b]%2==1:
                    self.data_burst_ext*=-1

                idx_start=np.around(eta_burst[b]/cst.PRI).astype(np.int)- cst.n_pulse_burst//2 + self.n_fft_az//2
                idx_end=idx_start+cst.n_pulse_burst

                if idx_start>=0 and idx_end<=self.n_fft_az:
                    
                    # IFFT to go back to the time domain
                    self.data[idx_start:idx_end,:]=np.fft.ifft(np.fft.fftshift(self.data_burst_ext,axes=1),axis=1)
                    
                    # Fine range migration and migration inside the burst + Doppler centroid correction
                    self.data[idx_start:idx_end,:]*=np.exp(2*1j*np.pi*(-cst.alpha*self.t[None,:]*2/cst.c*
                                                                       (rmc_fine[b]+self.radial_velocity*cst.eta_burst[:,None])
                                                                       +eta_pulse[b,:,None]*self.doppler_centroid))
                    if self.hamming_az:
                        self.data[idx_start:idx_end,:]*=self.hamming_window_az[:,None]

                    if self.hamming_range:
                        self.data[idx_start:idx_end,:]*=self.hamming_window_range[None,:]
                        
        # Azimuth FFT
        self.Data[:]=np.fft.fft(self.data,axis=0)
        
        # =============================================================================
        # Cmputation of the transfer function        
        # =============================================================================
        
        if update_h:
            
            # Range history
            # rs=np.sqrt((self.data_block.xs-x_focus)**2
            #            + (self.data_block.ys-y_focus)**2
            #            + (self.data_block.zs-z_focus)**2)
            
            # Doppler frequency history
            # fds=-2/cst.c*cst.fc*(self.data_block.vxs*(x_focus-self.data_block.xs)
            #                      + self.data_block.vys*(y_focus-self.data_block.ys)
            #                      + self.data_block.vzs*(z_focus-self.data_block.zs))/rs
        
            # Polynomial fit of the range and Doppler frequency history (order 4 and 1)
            coeffs_r=np.polyfit(eta_burst,rs,4)
            coeffs_fd=np.polyfit(eta_burst,fds,1)
            coeffs_dr=coeffs_r[:-1]*np.arange(4,0,-1)
            
            # Stationnary Phase Principle polynomial
            self.coeffs_poly_spp[:]=2/cst.c*(-cst.fc+cst.alpha*self.t_spp[None,:,None])*coeffs_dr[None,None,:]+1j*0
            self.coeffs_poly_spp[:,:,-1]+=-coeffs_fd[0]*self.t_spp[None,:]-self.fa_spp[:,None] + self.doppler_centroid - cst.alpha*self.t_spp[None,:]*2/cst.c*self.radial_velocity
        
            # Finding the roots (Cardan formulas)
            c_p=(3*self.coeffs_poly_spp[:,:,0]*self.coeffs_poly_spp[:,:,2]-self.coeffs_poly_spp[:,:,1]**2) / (3*self.coeffs_poly_spp[:,:,0]**2)
            c_q=(2*self.coeffs_poly_spp[:,:,1]**3 - 9*self.coeffs_poly_spp[:,:,0]*self.coeffs_poly_spp[:,:,1]*self.coeffs_poly_spp[:,:,2] + 27*self.coeffs_poly_spp[:,:,0]**2*self.coeffs_poly_spp[:,:,3]) / (27*self.coeffs_poly_spp[:,:,0]**3)
            delta1=c_q**2+4*c_p**3/27
            r1=(0.5*(-c_q-np.sqrt(delta1)))**(1.0/3) + (0.5*(-c_q+np.sqrt(delta1)))**(1.0/3) - self.coeffs_poly_spp[:,:,1]/(3*self.coeffs_poly_spp[:,:,0])
            delta2=(self.coeffs_poly_spp[:,:,1]+self.coeffs_poly_spp[:,:,0]*r1)**2 - 4*self.coeffs_poly_spp[:,:,0]*(self.coeffs_poly_spp[:,:,2] + r1*(self.coeffs_poly_spp[:,:,1] + r1*self.coeffs_poly_spp[:,:,0]))
            r3=np.real((-self.coeffs_poly_spp[:,:,1] - self.coeffs_poly_spp[:,:,0]*r1 + np.sqrt(delta2)) / (2*self.coeffs_poly_spp[:,:,0]))

            # Interpolation of the stationnary point
            f_interp_spp = interpolate.RectBivariateSpline(self.fa_spp, self.t_spp, r3, kx=3, ky=3)
            self.spp[:] = np.fft.fftshift(f_interp_spp(self.fa_shift,self.t),axes=0)
            
            # Computation od the transfer function
            fc,alpha,c,pi=cst.fc,cst.alpha,cst.c,np.pi
            cr0,cr1,cr2,cr3=coeffs_r[:-1]
            cfd0,cfd1=coeffs_fd
            fa_comp_h=self.fa[:,None]
            t_comp_h=self.t[None,:]
            spp_comp_h=self.spp
            radial_velocity=self.radial_velocity
            if use_numexpr:
                self.H[:]=numexpr.evaluate("exp(-2*1j*pi*((-fc+alpha*t_comp_h)*2/c*(cr0*spp_comp_h**4+cr1*spp_comp_h**3+cr2*spp_comp_h**2+(cr3-radial_velocity)*spp_comp_h)-(cfd0*spp_comp_h+cfd1)*t_comp_h-fa_comp_h*spp_comp_h))")
            else:
                self.H[:]=np.exp(-2*1j*pi*((-fc+alpha*t_comp_h)*2/c*(cr0*spp_comp_h**4+cr1*spp_comp_h**3+cr2*spp_comp_h**2+(cr3-radial_velocity)*spp_comp_h)-(cfd0*spp_comp_h+cfd1)*t_comp_h-fa_comp_h*spp_comp_h))
            
            if self.antenna_gain_compensation:
                self.H*=np.exp(2/cst.gamma*(fa_comp_h/(2*fc*self.velocity)*c)**2)
                
            # d2phi_spp==2*np.pi*2/c*(12*cr0*spp_comp_h**2+6*cr1*spp_comp_h+2*cr2)*(-fc+alpha*t_comp_h)
            d2phi_spp=2*np.pi*2/c*(2*cr2)*(-fc+alpha*t_comp_h)
            
            self.H*=np.sqrt(2*np.pi/np.abs(d2phi_spp)) / cst.PRI
        
        # Focusing the data in the frequency domain with a multiplication by the transfer function
        self.Data*=self.H
        
        # Applying delays
        if delay_range!=0 or delay_az!=0:
            self.Data*=np.exp(2*1j*np.pi*self.fa*delay_az)[:,None]
            self.Data*=np.exp(2*1j*np.pi*cst.alpha*self.t*2/cst.c*delay_range)[None,:]

        # Azumuth IFFT
        self.data[:]=np.fft.ifft(self.Data,axis=0)
        
        diff_eta=0*self.eta[self.n_fft_az//2-self.n_sl_output//2:self.n_fft_az//2+self.n_sl_output//2+self.n_sl_output%2]+np.repeat(self.eta[self.n_fft_az//2-self.n_sl_output//2+self.n_sl_output//self.n_multilook//2:self.n_fft_az//2+self.n_sl_output//2+self.n_sl_output%2:self.n_sl_output//self.n_multilook],self.n_sl_output//self.n_multilook)
        
        # Range FFT
        self.slc[:]=np.fft.fftshift(
            np.fft.fft(
            self.data[self.n_fft_az//2-self.n_sl_output//2:self.n_fft_az//2+self.n_sl_output//2+self.n_sl_output%2]*
            np.exp(2*1j*np.pi*cst.alpha*self.t[None,:]*2/cst.c*diff_eta[:,None]*self.radial_velocity)
            ,axis=1,n=cst.n_sample_range*self.range_ext_factor*self.zp)
            ,axes=1)[:,cst.n_sample_range*self.range_ext_factor//2*self.zp-cst.n_sample_range//2*self.zp:cst.n_sample_range*self.range_ext_factor//2*self.zp+cst.n_sample_range//2*self.zp]
        
        # Scaling
        self.slc[:]*=10**(-self.data_block.agc[self.n_burst//2]/20) / np.sqrt(cst.n_pulse_burst) / (cst.n_sample_range)
                     
        # Multilooking
        self.multilook[:]=np.mean(np.reshape(np.real(self.slc*np.conjugate(self.slc)),(self.n_multilook,-1,cst.n_sample_range*self.zp)),axis=1)
        self.msc[:]=np.abs(np.mean(np.reshape(self.slc,(self.n_multilook,-1,cst.n_sample_range*self.zp)),axis=1))**2 / self.multilook
        
        # Time tag of multilooks
        self.time_multilook[:]=self.data_block.time[self.n_burst//2]+self.eta[self.n_fft_az//2-self.n_sl_output//2+self.n_sl_output//self.n_multilook//2:self.n_fft_az//2+self.n_sl_output//2+self.n_sl_output%2:self.n_sl_output//self.n_multilook]
        
        # Interplation of the latitude and longitude of the multilooks
        f_lat=interpolate.interp1d(eta_burst,self.data_block.lat,kind="cubic")
        f_lon=interpolate.interp1d(eta_burst,np.unwrap(self.data_block.lon,discont=180),kind="cubic")
        self.lon_multilook[:]=f_lon(self.time_multilook-self.data_block.time[self.n_burst//2])
        self.lat_multilook[:]=f_lat(self.time_multilook-self.data_block.time[self.n_burst//2])
        
        # Adding the COG to the tracker
        self.tracker_multilook[:]=self.data_block.tracker[self.n_burst//2] + self.data_block.cog[self.n_burst//2]
        # self.alt_multilook[:]=self.data_block.alt[self.n_burst//2]   
        self.alt_multilook[:]=self.data_block.alt[self.n_burst//2] + self.radial_velocity*self.eta[self.n_fft_az//2-self.n_sl_output//2+self.n_sl_output//self.n_multilook//2:self.n_fft_az//2+self.n_sl_output//2+self.n_sl_output%2:self.n_sl_output//self.n_multilook]   
        
        # Scale factor
        self.scale_factor=self.data_block.agc[self.n_burst//2] + self.data_block.sig0_cal[self.n_burst//2] + 30*np.log10(self.data_block.alt[self.n_burst//2]) + 10*np.log10(1+self.data_block.alt[self.n_burst//2]/cst.geoid_wgs.a) + 10*np.log10(4*16*np.pi**2*cst.fc**2/cst.c**3/256/cst.Fs) -2*cst.GdB
        
        # Pulse peakiness
        self.pulse_peakiness[:]=np.max(self.multilook,axis=1)/np.mean(self.multilook,axis=1) * 85/128
        
        # =============================================================================
        # PLRM waveform computation and thermal noise estimation
        # =============================================================================
        
        # Range migration correction : radial velocity + Doppler correction
        self.rmc_plrm[:]=self.radial_velocity*(np.arange(-2,2)[:,None]*cst.BRI+cst.eta_burst[None,:]) - self.data_block.tracker[self.n_burst//2-2:self.n_burst//2+2:,None] + self.data_block.tracker[self.n_burst//2] - self.doppler_centroid*cst.c/2/cst.alpha 
        
        # Computation of the focusing operator
        self.h_plrm[:]=np.exp(-2*1j*np.pi*cst.alpha*cst.t[None,None,:]*2/cst.c*self.rmc_plrm[:,:,None])
        
        # Applying range migrations
        self.echo_foc_plrm[:]=self.data_block.echo[self.n_burst//2-2:self.n_burst//2+2,:,:]*self.h_plrm

        # Thermal noise estimation : the PLRM echo is computed with a Hamming window
        self.stack_complex_plrm[:]=np.fft.fftshift(np.fft.fft(self.echo_foc_plrm*self.hamming_window_range[None,None,::self.range_ext_factor],axis=2,n=cst.n_sample_range*self.zp),axes=2)
        self.stack_complex_plrm*=10**(-self.data_block.agc[self.n_burst//2]/20) / cst.n_sample_range
        self.stack_plrm[:]=self.stack_complex_plrm.real**2 + self.stack_complex_plrm.imag**2
        np.sum(self.stack_plrm,axis=(0,1),out=self.multilook_plrm)
        self.tn_plrm=np.mean(self.multilook_plrm[15*self.zp:20*self.zp])
        self.tn_ffsar=self.n_burst/256*self.tn_plrm
