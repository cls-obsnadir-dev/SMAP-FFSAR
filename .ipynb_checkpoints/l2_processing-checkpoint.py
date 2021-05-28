#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import leastsq
from scipy.signal import find_peaks

from cst import Cst

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

cst=Cst()

class OCOGretracker:
    """
    Classical OCOG retracker.
    """
    def __init__(self,zp=2,threshold=0.8,bias=0.0,use_msc=False):
        """
        Constructor of the class.
        """

        self.n_sample_range=cst.n_sample_range
        self.abs_ref_track=cst.abs_ref_track
        self.zp=zp
        self.threshold=threshold
        self.use_msc=use_msc
        self.tracker=np.nan
        self.scale_factor=np.nan
        self.bias=bias

        self.epoch=np.nan
        self.amplitude=np.nan
        if self.use_msc:
            self.msc_epoch=np.nan
        self.range=np.nan
        self.sig0=np.nan
        
        self.waveform=np.zeros(self.n_sample_range*self.zp,dtype=np.float)
        if self.use_msc:
            self.msc=np.zeros(self.n_sample_range*self.zp,dtype=np.float)

    def retrack_waveform(self,waveform,tracker,scale_factor,msc=None):
        """
        Retracks the waveform given in argument.
        """
        
        self.waveform[:]=waveform
        self.tracker=tracker
        self.scale_factor=scale_factor
        
        if self.use_msc:
            self.msc[:]=msc
        
        self.amplitude=np.sqrt(np.sum(self.waveform**4)/np.sum(self.waveform**2))

        i=np.where(self.waveform>self.threshold*self.amplitude)[0][0]

        if i==0:
            self.amplitude=np.nan
            self.epoch=np.nan
            if self.use_msc:
                self.msc_epoch=np.nan
            self.range=np.nan
            self.sig0=np.nan
        else:
            i_OCOG=i-1+(self.threshold*self.amplitude-self.waveform[i-1])/(self.waveform[i]-self.waveform[i-1])
            self.epoch=(i_OCOG/self.zp-self.abs_ref_track)/cst.B
            if self.use_msc:
                self.msc_epoch=np.interp(i_OCOG,np.arange(self.n_sample_range*self.zp),self.msc)
            self.range=self.epoch*cst.c/2+self.tracker
            self.sig0=10*np.log10(self.amplitude) + self.scale_factor + self.bias

    def get_data(self):
        """
        Returns estimated parameters.
        """

        dict_data={}
        dict_data['epoch_ocog']=self.epoch
        dict_data['amplitude_ocog']=self.amplitude
        if self.use_msc:
            dict_data['msc_epoch_ocog']=self.msc_epoch
        dict_data['range_ocog']=self.range
        dict_data['sig0_ocog']=self.sig0
            
        return dict_data
    
    def get_info(self):
        
        dict_variables={}
        
        dict_variables['epoch_ocog']={'type':np.float64,
                                      'attributes':{'units':'s',
                                                    'long_name':'Epoch from OCOG retracker'}}
        dict_variables['amplitude_ocog']={'type':np.float64,
                                          'attributes':{'units':'count',
                                                        'long_name':'Amplitude from OCOG retracker'}}
        if self.use_msc:
            dict_variables['msc_epoch_ocog']={'type':np.float64,
                                              'attributes':{'units':'None',
                                                            'long_name':'MSC of level 1b interpolated at the epoch from OCOG retracker'}}
        dict_variables['range_ocog']={'type':np.float64,
                                      'attributes':{'units':'m',
                                                    'long_name':'Range from OCOG retracker'}}
        dict_variables['sig0_ocog']={'type':np.float64,
                                     'attributes':{'units':'dB',
                                                   'long_name':'Sigma0 from OCOG retracker'}}
        return dict_variables

class PTRretracker:
    """
    Retracker with PTR (Point Target Response) model and least-square estimator.
    """  
    
    def __init__(self,zp=2,hamming_range=False,use_msc=False,sigma_mask=None):
        """
        Constructor of the class.
        """
        
        self.n_sample_range=cst.n_sample_range
        self.abs_ref_track=cst.abs_ref_track
        self.zp=zp
        self.hamming_range=hamming_range
        self.use_msc=use_msc
        self.tracker=np.nan
        self.scale_factor=np.nan
        
        self.fr=cst.t*cst.alpha

        # if self.hamming_range:
        #     self.window_range=(0.5-0.5*np.cos(2*np.pi*np.arange(0,cst.n_sample_range)/cst.n_sample_range))*np.exp(-2*1j*np.pi*self.fr*(self.n_sample_range//2-self.abs_ref_track)/cst.B)
        #     self.window_range*=1/np.sum(np.abs(self.window_range))*cst.n_sample_range
        # else:
        #     self.window_range=np.exp(-2*1j*np.pi*self.fr*(self.n_sample_range//2-self.abs_ref_track)/cst.B)
        
        if self.hamming_range:
            self.window_range=(cst.a1-cst.a2*np.cos(2*np.pi*np.arange(0,cst.n_sample_range)/cst.n_sample_range))*np.exp(-2*1j*np.pi*self.fr*(self.n_sample_range//2-self.abs_ref_track)/cst.B)
            self.window_range /= np.sqrt(cst.a1**2+0.5*cst.a2**2)
        else:
            self.window_range=np.exp(-2*1j*np.pi*self.fr*(self.n_sample_range//2-self.abs_ref_track)/cst.B)
            
        self.epoch=np.nan
        self.amplitude=np.nan
        if self.use_msc:
            self.msc_epoch=np.nan
        self.range=np.nan
        self.sig0=np.nan
        
        self.residual=np.zeros(self.n_sample_range*self.zp,dtype=np.float)
        
        if sigma_mask is not None:
            self.use_sigma_mask=True
            self.sigma_mask=sigma_mask / cst.B
            self.absc=(np.arange(0,self.n_sample_range,1/self.zp) - self.abs_ref_track)/cst.B
            self.mask_residual=np.zeros(self.n_sample_range*self.zp,dtype=np.float)
        else:
            self.use_sigma_mask=False
        
        self.waveform=np.zeros(self.n_sample_range*self.zp,dtype=np.float)
        if self.use_msc:
            self.msc=np.zeros(self.n_sample_range*self.zp,dtype=np.float)

    def residual_function(self,teta):
        
        epoch=teta[0]
        amplitude=teta[1]

        model = amplitude * np.abs(np.fft.fftshift(np.fft.fft(self.window_range*np.exp(2*1j*np.pi*self.fr*epoch),n=self.n_sample_range*self.zp)/self.n_sample_range))**2
        
        if self.use_sigma_mask:
            self.mask_residual[:]=np.exp(-(self.absc-epoch)**2/2/self.sigma_mask**2)
            residual = (model-self.waveform[:]) * self.mask_residual 
        else:
            residual = model-self.waveform[:]
        
        return residual

    def retrack_waveform(self,waveform,tracker,scale_factor,msc=None,teta_init=None,xtol=1e-5,ftol=1e-5):
        """
        Retracks the waveform given in argument.
        """
        
        self.waveform[:]=waveform
        self.tracker=tracker
        self.scale_factor=scale_factor
        
        if self.use_msc:
            self.msc[:]=msc

        if teta_init is None: 
            teta0=np.array([(np.argmax(self.waveform)/self.zp-self.abs_ref_track)/cst.B,np.max(self.waveform)])
        else:
            teta0=teta_init

        res_leastsq=leastsq(self.residual_function,teta0,full_output=False,xtol=xtol,ftol=ftol,diag=[5/cst.Fs,1000],epsfcn=1e-5,factor=10,maxfev=100)
        
        if res_leastsq[1] <=0 or res_leastsq[1] > 4:
            self.epoch=np.nan
            self.amplitude=np.nan
            self.residual[:]=np.nan
            self.range=np.nan
            self.sig0=np.nan
            if self.use_msc:
                self.msc_epoch=np.nan
        else:
            self.epoch=res_leastsq[0][0]
            self.amplitude=res_leastsq[0][1]
            if self.use_msc:
                self.msc_epoch=np.interp(self.epoch*cst.B+self.abs_ref_track,np.arange(self.n_sample_range*self.zp)/self.zp,self.msc)
            self.residual[:]=self.residual_function(res_leastsq[0])
            self.range=self.epoch*cst.c/2+self.tracker
            self.sig0=10*np.log10(self.amplitude)+scale_factor

    def get_data(self):
        """
        Returns estimated parameters.
        """

        dict_data={}
        dict_data['epoch_ptr']=self.epoch
        dict_data['amplitude_ptr']=self.amplitude
        if self.use_msc:
            dict_data['msc_epoch_ptr']=self.msc_epoch
        dict_data['range_ptr']=self.range
        dict_data['sig0_ptr']=self.sig0
        
        return dict_data
    
    def get_info(self):
        
        dict_variables={}
        
        dict_variables['epoch_ptr']={'type':np.float64,
                                      'attributes':{'units':'s',
                                                    'long_name':'Epoch from PTR retracker'}}
        dict_variables['amplitude_ptr']={'type':np.float64,
                                          'attributes':{'units':'count',
                                                        'long_name':'Amplitude from PTR retracker'}}
        if self.use_msc:
            dict_variables['msc_epoch_ptr']={'type':np.float64,
                                              'attributes':{'units':'None',
                                                            'long_name':'MSC of level 1b interpolated at the epoch from OCOG retracker'}}
        dict_variables['range_ptr']={'type':np.float64,
                                      'attributes':{'units':'m',
                                                    'long_name':'Range from PTR retracker'}}
        dict_variables['sig0_ptr']={'type':np.float64,
                                     'attributes':{'units':'dB',
                                                   'long_name':'Sigma0 from PTR retracker'}}
        
        return dict_variables

class MultiPTRretracker:
    """
    Retracker with PTR (Point Target Response) model and least-square estimator.
    """  
    
    def __init__(self,zp=2,n_estimates=3,hamming_range=False,use_msc=False,sigma_mask=None):
        """
        Constructor of the class.
        """
        
        self.n_sample_range=cst.n_sample_range
        self.abs_ref_track=cst.abs_ref_track
        self.zp=zp
        self.hamming_range=hamming_range
        self.use_msc=use_msc
        self.tracker=np.nan
        self.scale_factor=np.nan
        self.n_estimates=n_estimates
        
        self.waveform=np.zeros(self.n_sample_range*self.zp,dtype=np.float)
        if self.use_msc:
            self.msc=np.zeros(self.n_sample_range*self.zp,dtype=np.float)
            
        self.retracker_PTR=PTRretracker(self.zp,self.hamming_range,self.use_msc,sigma_mask=sigma_mask)
        
        self.epoch=np.nan*np.ones(self.n_estimates)
        self.amplitude=np.nan*np.ones(self.n_estimates)
        if self.use_msc:
            self.msc_epoch=np.nan*np.ones(self.n_estimates)
        self.range=np.nan*np.ones(self.n_estimates)
        self.sig0=np.nan*np.ones(self.n_estimates)

    def retrack_waveform(self,waveform,tracker,scale_factor,msc=None):
        """
        Retracks the waveform given in argument.
        """
        
        self.waveform[:]=waveform
        self.tracker=tracker
        self.scale_factor=scale_factor
        
        if self.use_msc:
            self.msc[:]=msc
            
        try:
            peak_absc,peak_heights=find_peaks(self.waveform,distance=2*self.zp,height=0)
            ind_sort=np.argsort(peak_heights['peak_heights'])
        except:
            self.epoch=np.nan*np.ones(self.n_estimates)
            self.amplitude=np.nan*np.ones(self.n_estimates)
            if self.use_msc:
                self.msc_epoch=np.nan*np.ones(self.n_estimates)
            self.range=np.nan*np.ones(self.n_estimates)
            self.sig0=np.nan*np.ones(self.n_estimates)
        else:
            for i in range(1,self.n_estimates+1):
            
                teta0=np.array([(peak_absc[ind_sort][-i]/self.zp-self.abs_ref_track)/cst.B,peak_heights['peak_heights'][ind_sort][-i]])
                
                if self.use_msc:
                    self.retracker_PTR.retrack_waveform(self.waveform, self.tracker, self.scale_factor,msc=self.msc,teta_init=teta0)
                else:
                    self.retracker_PTR.retrack_waveform(self.waveform, self.tracker, self.scale_factor,teta_init=teta0)
                
                self.epoch[i-1]=self.retracker_PTR.epoch
                self.amplitude[i-1]=self.retracker_PTR.amplitude
                self.range[i-1]=self.retracker_PTR.range
                self.sig0[i-1]=self.retracker_PTR.sig0
                if self.use_msc:
                    self.msc_epoch[i-1]=self.retracker_PTR.msc_epoch
    
            sig0_sort=np.copy(self.sig0)
            sig0_sort[np.isnan(sig0_sort)]=-np.inf
            ind_sort=np.argsort(sig0_sort)[::-1]
            
            self.epoch=self.epoch[ind_sort]
            self.amplitude=self.amplitude[ind_sort]
            self.sig0=self.sig0[ind_sort]
            self.range=self.range[ind_sort]
            if self.use_msc:
                self.msc_epoch=self.msc_epoch[ind_sort]

    def get_data(self):
        """
        Returns estimated parameters.
        """

        dict_data={}
        
        for i in range(1,self.n_estimates+1):
            dict_data['epoch_multiptr'+str(i)]=self.epoch[i-1]
            dict_data['amplitude_multiptr'+str(i)]=self.amplitude[i-1]
            if self.use_msc:
                dict_data['msc_epoch_multiptr'+str(i)]=self.msc_epoch[i-1]
            dict_data['range_multiptr'+str(i)]=self.range[i-1]
            dict_data['sig0_multiptr'+str(i)]=self.sig0[i-1]
        
        return dict_data
    
    def get_info(self):
        
        dict_variables={}
        
        for i in range(1,self.n_estimates+1):
        
            dict_variables['epoch_multiptr'+str(i)]={'type':np.float64,
                                               'attributes':{'units':'s',
                                                             'long_name':'Epoch from multi PTR retracker '+str(i)}}
            dict_variables['amplitude_multiptr'+str(i)]={'type':np.float64,
                                                   'attributes':{'units':'count',
                                                                 'long_name':'Amplitude from multi PTR retracker '+str(i)}}
            if self.use_msc:
                dict_variables['msc_epoch_multiptr'+str(i)]={'type':np.float64,
                                                       'attributes':{'units':'None',
                                                                     'long_name':'MSC of level 1b interpolated at the epoch from OCOG retracker '+str(i)}}
            dict_variables['range_multiptr'+str(i)]={'type':np.float64,
                                               'attributes':{'units':'m',
                                                             'long_name':'Range from multi PTR retracker '+str(i)}}
            dict_variables['sig0_multiptr'+str(i)]={'type':np.float64,
                                              'attributes':{'units':'dB',
                                                            'long_name':'Sigma0 from multi PTR retracker '+str(i)}}
        
        return dict_variables
