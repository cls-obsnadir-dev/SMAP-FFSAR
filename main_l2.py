#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import configparser
import logging
from datetime import datetime
import netCDF4
import numpy as np

from ffsar_processing import BackprojectionProcessing
from l2_processing import OCOGretracker, PTRretracker, MultiPTRretracker
from cst import Cst
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

cst=Cst()

# =============================================================================
# Parsing arguments
# =============================================================================
parser=argparse.ArgumentParser()
parser.add_argument("input_file",help="path to L1b netCDF4 input file")
parser.add_argument("-o","--output_file",help="path to L2 netCDF4 output file")
parser.add_argument("-c","--config_file",help="path to .cfg config file")
parser.add_argument("-l","--log",help="path to log file")
args = parser.parse_args()

# =============================================================================
# Opening L1b netCDF4 file
# =============================================================================
try:
    file_l1b=netCDF4.Dataset(args.input_file,"r")
except:
    sys.stderr.write("Unable to open/read L1b input file.\n")
    sys.exit(1)

# =============================================================================
# The name of the product (ending .SEN3 is removed)
# =============================================================================
basename=file_l1b.product_name.split(".")[0]

# =============================================================================
# Setting args.config_file to its default value
# =============================================================================
current_file=__file__
if args.config_file is None:
    args.config_file=os.path.join(os.path.dirname(os.path.realpath(current_file)),"config.cfg")

# =============================================================================
# Opening the config file
# =============================================================================
config=configparser.ConfigParser()
try:
    config.read(args.config_file)
except:
    sys.stderr.write("Error: unable to open/read config file.\n")
    sys.exit(1)

# =============================================================================
# Output directory
# =============================================================================
dir_out=os.path.realpath(config.get('DIR','output_dir'))

# =============================================================================
# Output file
# =============================================================================
if args.output_file is None:
    dir_out_l2=os.path.join(dir_out,"l2")
    os.makedirs(dir_out_l2,exist_ok=True)
    args.output_file=os.path.join(dir_out_l2,basename+"_l2.nc")
# remove existing output file
if os.path.exists(args.output_file):
    os.remove(args.output_file)

# =============================================================================
# Log file
# =============================================================================
if args.log is None:
    dir_out_log=os.path.join(dir_out,"log")
    os.makedirs(dir_out_log,exist_ok=True)
    args.log=os.path.join(dir_out_log,basename+"_l2.log")

for handler in logging.root.handlers:
    logging.root.removeHandler(handler)
logging.basicConfig(filename=args.log,level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',datefmt='%m/%d/%Y %I:%M:%S %p',filemode='w')
logging.info("Starting processing product {}".format(basename))

# =============================================================================
# Opening L2 netCDF4 output file
# =============================================================================
try:
    file_l2=netCDF4.Dataset(args.output_file,"w")
except:
    logging.critical("Unable to open L2 output file.")
    sys.exit(1)
else:
    logging.info("Creating output file {}".format(args.output_file))

utcnow=datetime.utcnow() 
file_l2.creation_time="{}-{}-{}T{}:{}:{}Z".format(str(utcnow.year).zfill(4),str(utcnow.month).zfill(2),str(utcnow.day).zfill(2),\
                                                   str(utcnow.hour).zfill(2),str(utcnow.minute).zfill(2),str(utcnow.second).zfill(2))

# =============================================================================
# Reading configuration file
# =============================================================================
l1b_processing=config.get('PROCESSING','l1b_processing')
l2_processing=config.get('PROCESSING','l2_processing').split(",")
zp=int(config.get('PROCESSING_OPTIONS','zp'))
illumination_time=float(config.get('PROCESSING_OPTIONS','illumination_time'))
n_looks=int(illumination_time/cst.BRI)
if config.get('PROCESSING_OPTIONS', 'hamming_range') == "yes":
    hamming_range = True
elif config.get('PROCESSING_OPTIONS', 'hamming_range') == "no":
    hamming_range = False

if config.get('PROCESSING_OPTIONS', 'hamming_az') == "yes":
    hamming_az = True
elif config.get('PROCESSING_OPTIONS', 'hamming_az') == "no":
    hamming_az = False

if float(config.get('PROCESSING_OPTIONS','OCOG_threshold_sar'))>=0.0 and float(config.get('PROCESSING_OPTIONS','OCOG_threshold_sar'))<=1.0:
    OCOG_threshold_sar=float(config.get('PROCESSING_OPTIONS','OCOG_threshold_sar'))
else:
    logging.error("Uncorrect OCOG_threshold_sar, must be between 0.0 and 1.0")
    sys.exit(0)

if int(config.get('PROCESSING_OPTIONS','MultiPTR_n_estimates'))>=1 and int(config.get('PROCESSING_OPTIONS','MultiPTR_n_estimates'))<=10:
    MultiPTR_n_estimates=int(config.get('PROCESSING_OPTIONS','MultiPTR_n_estimates'))
else:
    logging.error("Uncorrect MultiPTR_n_estimates, must be between 1 and 10")
    sys.exit(0)

# =============================================================================
# Defining variable names
# =============================================================================
if l1b_processing=="ffsar_bp":
    suffix=BackprojectionProcessing.suffix
    
name_dim_time="time"+suffix
name_var_time="time"+suffix
name_var_waveform="multilook"+suffix
name_var_waveform_plrm="multilook_plrm"
name_var_rip="rip"+suffix
name_var_stack="stack"+suffix
name_var_tracker="tracker"+suffix
name_var_scale_factor="scale_factor"+suffix
name_var_lon="lon"+suffix
name_var_lat="lat"+suffix

name_var_alt="alt"+suffix
name_var_vs="velocity"+suffix
name_var_tn="tn"+suffix
name_var_tn_plrm="tn_plrm"
name_var_rmc_mask="rmc_coarse"+suffix
name_var_doppler_freqs="doppler_freqs"+suffix

if suffix=="_ffsar":
    use_msc=True
else:
    use_msc=False

# =============================================================================
# Defining objects for processing
# =============================================================================
if 'OCOG_SAR' in l2_processing:
    retracker_OCOG_sar=OCOGretracker(zp=zp,threshold=OCOG_threshold_sar,use_msc=use_msc,bias=cst.sig0_bias_ocog_sar)
    logging.info("Applying the OCOG SAR retracker")
if 'PTR' in l2_processing:
    retracker_PTR=PTRretracker(zp=zp,hamming_range=hamming_range,use_msc=use_msc,sigma_mask=None)
    logging.info("Applying the PTR retracker")
if 'MultiPTR' in l2_processing:
    retracker_MultiPTR=MultiPTRretracker(zp=zp,n_estimates=MultiPTR_n_estimates,hamming_range=hamming_range,use_msc=use_msc,sigma_mask=None)
    logging.info("Applying the MultiPTR retracker")

# =============================================================================
# Initializing the l2 netCDF file
# =============================================================================
n_l1b_records=file_l1b.dimensions[name_dim_time].size
list_var_l1b=list(file_l1b.variables.keys())
list_var_l2=[]

# Dimensions
file_l2.createDimension(name_dim_time,n_l1b_records)

# Attributes
file_l2.product_name=file_l1b.product_name
file_l2.mission_name=file_l1b.mission_name
file_l2.software_version=file_l1b.software_version
file_l2.producer=file_l1b.producer
file_l2.project_website=file_l1b.project_website
file_l2.Conventions=file_l1b.Conventions
file_l2.reference_ellipsoid=file_l1b.reference_ellipsoid
file_l2.semi_major_ellipsoid_axis=file_l1b.semi_major_ellipsoid_axis
file_l2.ellipsoid_flattening=file_l1b.ellipsoid_flattening

# More attributes
file_l2.first_meas_time=file_l1b.first_meas_time
file_l2.last_meas_time=file_l1b.last_meas_time
file_l2.first_meas_lat=file_l1b.first_meas_lat
file_l2.last_meas_lat=file_l1b.last_meas_lat
file_l2.first_meas_lon=file_l1b.first_meas_lon
file_l2.last_meas_lon=file_l1b.last_meas_lon
file_l2.cycle_number=file_l1b.cycle_number
file_l2.pass_number=file_l1b.pass_number

# Geographical selection
file_l2.shapefile_path=file_l1b.shapefile_path
file_l2.bbox_lat_min=file_l1b.bbox_lat_min
file_l2.bbox_lat_max=file_l1b.bbox_lat_max
file_l2.bbox_lon_min=file_l1b.bbox_lon_min
file_l2.bbox_lon_max=file_l1b.bbox_lon_max
file_l2.surface_type=file_l1b.surface_type

# Configuration parameters
file_l2.l1b_processing=file_l1b.l1b_processing
file_l2.illumination_time=file_l1b.illumination_time
file_l2.posting_rate=file_l1b.posting_rate
file_l2.range_ext_factor=file_l1b.range_ext_factor
file_l2.zp=file_l1b.zp
file_l2.hamming_az=file_l1b.hamming_az
file_l2.hamming_range=file_l1b.hamming_range
file_l2.l2_processing=l2_processing
if 'OCOG_SAR' in l2_processing:
    file_l2.OCOG_threshold_sar=OCOG_threshold_sar
if 'MultiPTR' in l2_processing:
    file_l2.MultiPTR_n_estimates=MultiPTR_n_estimates

# Existing variables from l1b
for k in list_var_l1b:
    if len(file_l1b.variables[k].dimensions)==1 and file_l1b.variables[k].dimensions[0]==name_dim_time:
        file_l2.createVariable(k,file_l1b.variables[k].dtype,dimensions=(name_dim_time))
        file_l2.variables[k][:]=file_l1b.variables[k][:]
        file_l2.variables[k].setncatts(file_l1b.variables[k].__dict__)
        list_var_l2.append(k)

# New variables
if 'OCOG_SAR' in l2_processing:
    dict_var_OCOG_SAR=retracker_OCOG_sar.get_info()
    for v in dict_var_OCOG_SAR:
        file_l2.createVariable(v+suffix,dict_var_OCOG_SAR[v]['type'],dimensions=name_dim_time)
        file_l2.variables[v+suffix].setncatts(dict_var_OCOG_SAR[v]['attributes'])
        file_l2.variables.dimensions=name_var_lon+" "+name_var_lat
        list_var_l2.append(v+suffix)

if 'PTR' in l2_processing:
    dict_var_PTR=retracker_PTR.get_info()
    for v in dict_var_PTR:
        file_l2.createVariable(v+suffix,dict_var_PTR[v]['type'],dimensions=name_dim_time)
        file_l2.variables[v+suffix].setncatts(dict_var_PTR[v]['attributes'])
        file_l2.variables.dimensions=name_var_lon+" "+name_var_lat
        list_var_l2.append(v+suffix)
        
if 'MultiPTR' in l2_processing:
    dict_var_MultiPTR=retracker_MultiPTR.get_info()
    for v in dict_var_MultiPTR:
        file_l2.createVariable(v+suffix,dict_var_MultiPTR[v]['type'],dimensions=name_dim_time)
        file_l2.variables[v+suffix].setncatts(dict_var_MultiPTR[v]['attributes'])
        file_l2.variables.dimensions=name_var_lon+" "+name_var_lat
        list_var_l2.append(v+suffix)

# =============================================================================
# Reading necessary information from L1b file : reading all at once, might be too big ...
# =============================================================================
waveform=file_l1b.variables[name_var_waveform][:]
tracker=file_l1b.variables[name_var_tracker][:]
scale_factor=file_l1b.variables[name_var_scale_factor][:]
alt=file_l1b.variables[name_var_alt][:]
lat=file_l1b.variables[name_var_lat][:]
vs=file_l1b.variables[name_var_vs][:]
tn=file_l1b.variables[name_var_tn][:]
rt=np.sqrt(cst.geoid_wgs.a**2*np.cos(np.radians(lat))**2 + cst.geoid_wgs.b**2*np.sin(np.radians(lat))**2)
rmc_mask=file_l1b.variables[name_var_rmc_mask][:]
doppler_freqs=file_l1b.variables[name_var_doppler_freqs][:]
msc=file_l1b.variables["msc_ffsar"][:]

# =============================================================================
# Performing the processing
# =============================================================================
l1b_bar = np.arange(n_l1b_records)

for i in tqdm(l1b_bar, desc="L2"):

    logging.debug("Retracking record {}".format(i))

    # =============================================================================
    # OCOG SAR retracker
    # =============================================================================
    if 'OCOG_SAR' in l2_processing:
        if suffix=="_ffsar":
            retracker_OCOG_sar.retrack_waveform(waveform[i,:],tracker[i],scale_factor[i],msc[i,:])
        else:
            retracker_OCOG_sar.retrack_waveform(waveform[i,:],tracker[i],scale_factor[i])
    
        dict_data_OCOG_SAR=retracker_OCOG_sar.get_data()
        for v in dict_data_OCOG_SAR:
            file_l2.variables[v+suffix][i]=dict_data_OCOG_SAR[v]
    
    # =============================================================================
    # PTR retracker
    # =============================================================================
    if 'PTR' in l2_processing:
        if suffix=="_ffsar":
            retracker_PTR.retrack_waveform(waveform[i,:],tracker[i],scale_factor[i],msc[i,:])
        else:
            retracker_PTR.retrack_waveform(waveform[i,:],tracker[i],scale_factor[i])
    
        dict_data_PTR=retracker_PTR.get_data()
        for v in dict_data_PTR:
            file_l2.variables[v+suffix][i]=dict_data_PTR[v]
            
    # =============================================================================
    # Multi PTR retracker
    # =============================================================================
    if 'MultiPTR' in l2_processing:
        if suffix=="_ffsar":
            retracker_MultiPTR.retrack_waveform(waveform[i,:],tracker[i],scale_factor[i],msc[i,:])
        else:
            retracker_MultiPTR.retrack_waveform(waveform[i,:],tracker[i],scale_factor[i])
    
        dict_data_MultiPTR=retracker_MultiPTR.get_data()
        for v in dict_data_MultiPTR:
            file_l2.variables[v+suffix][i]=dict_data_MultiPTR[v]
            
# =============================================================================
# Closing netCDF files
# =============================================================================
logging.info("L2 processing done. Closing L1b and L2 files.")
file_l1b.close()
file_l2.close()