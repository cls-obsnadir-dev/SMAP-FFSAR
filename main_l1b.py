#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
import sys
import argparse
import configparser
from datetime import datetime,timedelta

import numpy as np
import netCDF4
import shapefile

from l1b_processing import DataBuffer,DataBlock
from ffsar_processing import BackprojectionProcessing
from cst import Cst
import utils
from platform import python_version
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

cst=Cst()
date0=datetime(2000,1,1)

# =============================================================================
# Parsing arguments
# =============================================================================
parser=argparse.ArgumentParser()
parser.add_argument("input_file",help="path to L1a netCDF4 input file")
parser.add_argument("-o","--output_file",help="path to L1b netCDF4 output file")
parser.add_argument("-c","--config_file",help="path to .cfg config file")
parser.add_argument("-l","--log",help="path to log file")
parser.add_argument("-s","--start",help="date for start of processing YYYYMMDDTHHMMSS")
parser.add_argument("-e","--end",help="date for end of processing YYYYMMDDTHHMMSS")
args = parser.parse_args()

# =============================================================================
# Opening L1a netCDF4 file
# =============================================================================
try:
    file_l1a=netCDF4.Dataset(args.input_file,"r")
except:
    sys.stderr.write("Unable to open/read L1a input file.\n")
    sys.exit(1)

# =============================================================================
# The name of the product (ending .SEN3 is removed)
# =============================================================================
basename=file_l1a.product_name.split(".")[0]

# =============================================================================
# Setting args.config_file to its default value if not specified (the config.cfg 
# located in the same directory as this file is used)
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
    dir_out_l1b=os.path.join(dir_out,"l1b")
    os.makedirs(dir_out_l1b,exist_ok=True)
    args.output_file=os.path.join(dir_out_l1b,basename+"_l1b.nc")
# remove existing output file
if os.path.exists(args.output_file):
    os.remove(args.output_file)

# =============================================================================
# Log file
# =============================================================================
if args.log is None:
    dir_out_log=os.path.join(dir_out,"log")
    os.makedirs(dir_out_log,exist_ok=True)
    args.log=os.path.join(dir_out_log,basename+"_l1b.log")

for handler in logging.root.handlers:
    logging.root.removeHandler(handler)
logging.basicConfig(filename=args.log,level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',datefmt='%m/%d/%Y %I:%M:%S %p',filemode='w')
logging.info("Starting processing product {}".format(basename))

# =============================================================================
# Start/end dates for processing
# =============================================================================
if args.start is not None:
    date_start=datetime(int(args.start[:4]), int(args.start[4:6]), int(args.start[6:8]),int(args.start[9:11]),int(args.start[11:13]),int(args.start[13:15]))
    use_date_start=True
    logging.info("Processing from {}.".format(args.start))
else:
    use_date_start=False
if args.end is not None:
    date_end=datetime(int(args.end[:4]), int(args.end[4:6]), int(args.end[6:8]),int(args.end[9:11]),int(args.end[11:13]),int(args.end[13:15]))
    use_date_end=True
    logging.info("Processing until {}.".format(args.end))
else:
    use_date_end=False

# =============================================================================
# Bounding box
# =============================================================================
try:
    bbox_lat_min=float(config.get("GENERAL_PARAMETERS","lat_min"))
    bbox_lat_max=float(config.get("GENERAL_PARAMETERS","lat_max"))
    bbox_lon_min=float(config.get("GENERAL_PARAMETERS","lon_min"))
    bbox_lon_max=float(config.get("GENERAL_PARAMETERS","lon_max"))
except:
    logging.info("No correct bbox information, continuing without bbox.")
    use_bbox=False
else:
    logging.info("Using bbox : {:.2f} < lat < {:.2f} & {:.2f} < lon < {:.2f}".format(bbox_lat_min,bbox_lat_max,bbox_lon_min,bbox_lon_max))
    use_bbox=True
    
# =============================================================================
# Surface type
# =============================================================================
try:
    selected_surf_type=[int(e) for e in config.get('GENERAL_PARAMETERS','surf_type').split(",")]
except:
    logging.info("No correct surface type information, continuing without surface type.")
    use_surf_type=False
else:
    logging.info("Using surface type : {}".format(",".join([str(e) for e in selected_surf_type])))
    use_surf_type=True

# =============================================================================
# Opening the shapefile
# =============================================================================
shapefile_path=config.get("FILES","shapefile")
if shapefile_path != "":
    try:
        sh=shapefile.Reader(os.path.realpath(shapefile_path))
    except:
        logging.error("Unable to read shapefile, continuing without shapefile")
        use_shapefile=False
    else:
        logging.info("Using shapefile {}".format(os.path.realpath(shapefile_path)))
        use_shapefile=True
else:
    use_shapefile=False

# =============================================================================
# Computing cycle/track information consitent with shapefile data 
# =============================================================================
nb_cycle=int(basename[69:72])
nb_orbit=int(basename[73:76])
latitude=file_l1a.variables["lat_l1a_echo_sar_ku"][:]

cycle_number_main,cycle_number,pass_number_main,pass_number=utils.compute_cycle_pass(nb_cycle,nb_orbit,latitude)
set_pass_number=list(set(pass_number))

# =============================================================================
# Opening CAL2 ASCII file
# =============================================================================
try:
    with open(os.path.realpath(config.get('FILES','cal2_file')),"r") as file_cal2:
        gprw_mean=np.loadtxt(file_cal2)
    # It needs to be rolled by one, because the zero frequency of the GPRW is index 63 (starting from 0), whereas numpy FFTs return the zero frequency for index 64    
    gprw_mean=np.roll(gprw_mean,1)
except:
    logging.critical("Unable to open CAL2 file")
    sys.exit(1)
else:
    logging.info("Using CAL2 file : {}".format(os.path.realpath(config.get('FILES','cal2_file'))))

# =============================================================================
# Selecting useful (relative to current pass) polygons in the shapefile
# =============================================================================
if use_shapefile:
    shapefile_records=sh.records()
    ind_shapes=np.where([(r.pass_nb in set_pass_number) for r in shapefile_records])[0]
    n_shapes=ind_shapes.size
    logging.info("The pass has {} associated shape(s) in the shapefile".format(n_shapes))
    if n_shapes>0:
        shapefile_shapes=[sh.shape(i) for i in ind_shapes]
        shapefile_records=[sh.record(i) for i in ind_shapes]
        shapefile_bbox=np.array([e.bbox for e in shapefile_shapes])
    else:
        logging.error("No pass number of the shapefile corresponding")
        sys.exit(0)
    sh.close()

# =============================================================================
# Selecting 20 Hz radar cycles to process
# =============================================================================
burst_time=file_l1a.variables["time_l1a_echo_sar_ku"][:]
burst_latitude=file_l1a.variables["lat_l1a_echo_sar_ku"][:]
burst_longitude=file_l1a.variables["lon_l1a_echo_sar_ku"][:]
burst_count_cycle=file_l1a.variables["burst_count_cycle_l1a_echo_sar_ku"][:]
surf_type=file_l1a.variables["surf_type_l1a_echo_sar_ku"][:]
burst_time=file_l1a.variables['time_l1a_echo_sar_ku'][:]
burst_size = len(burst_time)

if use_bbox:
    test_bbox=(burst_latitude<=bbox_lat_max) & \
        (burst_latitude>=bbox_lat_min) & \
        (burst_longitude<=bbox_lon_max) & \
        (burst_longitude>=bbox_lon_min)
else:
    test_bbox=np.empty(burst_latitude.size,dtype=np.bool)
    test_bbox[:]=True

if use_shapefile:
    burst_longitude_bis=np.copy(burst_longitude)
    burst_longitude_bis[burst_longitude>180]+=-360
    # Coarse mean we only use the shapefile bbox (not the polygon)
    test_shapefile_coarse=np.any((burst_latitude[:,None]>=shapefile_bbox[None,:,1]) & \
                                 (burst_latitude[:,None]<=shapefile_bbox[None,:,3]) & \
                                 (burst_longitude_bis[:,None]>=shapefile_bbox[None,:,0]) & \
                                 (burst_longitude_bis[:,None]<=shapefile_bbox[None,:,2]),axis=1)
else:
    test_shapefile_coarse=np.empty(burst_latitude.size,dtype=np.bool)
    test_shapefile_coarse[:]=True
    
if use_surf_type:
    test_surf_type=np.array([e in selected_surf_type for e in surf_type])
else:
    test_surf_type=np.empty(burst_latitude.size,dtype=np.bool)
    test_surf_type[:]=True
    
if use_date_start:
    test_date_start=burst_time>(date_start-date0).total_seconds()
else:
    test_date_start=np.empty(burst_latitude.size,dtype=np.bool)
    test_date_start[:]=True

if use_date_end:
    test_date_end=burst_time<(date_end-date0).total_seconds()
else:
    test_date_end=np.empty(burst_latitude.size,dtype=np.bool)
    test_date_end[:]=True

test_burst=test_bbox & test_shapefile_coarse & test_surf_type & test_date_start & test_date_end
ind_burst_process=np.where(test_burst)[0]

# If no burst is to be processed
if ind_burst_process.size==0:
    logging.info("No data has to be processed. Ending the processing.")
    sys.exit(0)
else:
    logging.info("{} bursts satisfy selection.".format(ind_burst_process.size))

# =============================================================================
# Opening L1b netCDF4 output file
# =============================================================================
try:
    file_l1b=netCDF4.Dataset(args.output_file,"w")
except:
    logging.critical("Unable to open L1b output file.")
    sys.exit(1)
else:
    logging.info("Creating output file {}".format(args.output_file))

utcnow=datetime.utcnow() 
file_l1b.creation_time="{}-{}-{}T{}:{}:{}Z".format(str(utcnow.year).zfill(4),str(utcnow.month).zfill(2),str(utcnow.day).zfill(2),\
                                                   str(utcnow.hour).zfill(2),str(utcnow.minute).zfill(2),str(utcnow.second).zfill(2))

# =============================================================================
# Defining objects for processing
# =============================================================================

# by default values
n_burst_shift=4
n_sl_output=906

# values read in the configuration file
l1b_processing=config.get('PROCESSING','l1b_processing')
if float(config.get('PROCESSING_OPTIONS','illumination_time'))>=0.08 and float(config.get('PROCESSING_OPTIONS','illumination_time'))<=2.3:
    illumination_time=float(config.get('PROCESSING_OPTIONS','illumination_time'))
else:
    logging.error("Uncorrect illumination_time, must be between 0.08 and 2.3")
    sys.exit(0)
    
if float(config.get('PROCESSING_OPTIONS','posting_rate'))>=20 and float(config.get('PROCESSING_OPTIONS','posting_rate'))<=17825:
    posting_rate=float(config.get('PROCESSING_OPTIONS','posting_rate'))
else:
    logging.error("Uncorrect posting_rate, must be between 20 and 17825")
    sys.exit(0)
    
if int(config.get('PROCESSING_OPTIONS','zp'))>=1 and int(config.get('PROCESSING_OPTIONS','zp'))<=2:
    zp=int(config.get('PROCESSING_OPTIONS','zp'))
else:
    logging.error("Uncorrect zp, must be equal to 1 or 2")
    sys.exit(0)
    
if int(config.get('PROCESSING_OPTIONS','range_ext_factor'))>=1 and int(config.get('PROCESSING_OPTIONS','range_ext_factor'))<=2:
    range_ext_factor=int(config.get('PROCESSING_OPTIONS','range_ext_factor'))
else:
    logging.error("Uncorrect range_ext_factor, must be equal to 1 or 2")
    sys.exit(0)
    
if config.get('PROCESSING_OPTIONS','hamming_range')=="yes":
    hamming_range=True
elif config.get('PROCESSING_OPTIONS','hamming_range')=="no":
    hamming_range=False
else:
    logging.error("Uncorrect hamming_range, must be `yes` or `no`")
    sys.exit(0)
    
if config.get('PROCESSING_OPTIONS','hamming_az')=="yes":
    hamming_az=True
elif config.get('PROCESSING_OPTIONS','hamming_az')=="no":
    hamming_az=False
else:
    logging.error("Uncorrect hamming_az, must be `yes` or `no`")
    sys.exit(0)

# calculating the number of single-looks (sl) and multi-looks (ml), so that ml is a divisor of sl
n_multilook = int(n_sl_output*posting_rate*cst.PRI)
n_sl_output = int(n_sl_output/n_multilook)*n_multilook

# convert into pulses or bursts
n_burst_block=int(illumination_time/cst.BRI)

data_buffer=DataBuffer(file_l1a,n_burst_block)
data_block=DataBlock(n_burst_block,data_buffer,gprw_mean)

if l1b_processing=="ffsar_bp":
    processing=BackprojectionProcessing(data_block, n_sl_output,n_multilook=n_multilook,hamming_az=hamming_az,hamming_range=hamming_range,zp=zp,range_ext_factor=range_ext_factor)
    logging.info("Using the FF-SAR back-projection processing")
else:
    logging.error("Uncorrect l1b processing.")
    sys.exit(0)
    
# =============================================================================
# Initializing the l1b netCDF file
# =============================================================================

dict_dimensions,dict_variables,suffix=processing.get_info()
list_dimensions=dict_dimensions.keys()
list_variables=dict_variables.keys()

name_dim_time="time"+suffix
name_var_time="time"+suffix
name_var_lat="lat"+suffix
name_var_lon="lon"+suffix

# Dimensions
for k in list_dimensions:
    file_l1b.createDimension(k,dict_dimensions[k])

# Attributes
file_l1b.product_name=file_l1a.product_name
file_l1b.mission_name=file_l1a.mission_name
file_l1b.software_version="v3.0"
file_l1b.producer="CLS"
file_l1b.Conventions="CF-1.6"
file_l1b.reference_ellipsoid="WGS84"
file_l1b.semi_major_ellipsoid_axis=cst.geoid_wgs.a
file_l1b.ellipsoid_flattening=cst.geoid_wgs.f

# Variables
for k in list_variables:
    file_l1b.createVariable(k,dict_variables[k]['type'],dimensions=dict_variables[k]['dimension'])
    file_l1b.variables[k].setncatts(dict_variables[k]['attributes'])
    file_l1b.set_auto_scale(True)
    if "coordinates" in dict_variables[k].keys():
        file_l1b.variables[k].coordinates=" ".join(dict_variables[k]['coordinates'])

# =============================================================================
# Performing the processing
# =============================================================================

previous_burst_time=0
index_write_nc=0

burst_bar = ind_burst_process[::n_burst_shift]

for b in tqdm(burst_bar, desc="L1b"):
    
    logging.debug("Processing block centered on burst {}".format(b))
    
    deltat=burst_time[b]-previous_burst_time

    # Checking whether the data buffer needs updating to process burst b is available in the data buffer
    if b > data_buffer.index_end - n_burst_block//2 or np.isnan(data_buffer.index_end):
        data_buffer_start=max(0,b-n_burst_block//2)
        data_buffer_end=min(burst_size,b+n_burst_block//2)
        data_buffer.load(data_buffer_start,data_buffer_end)

    # Loading data block
    data_block.load(b-data_buffer_start)

    # Processing the data
    if l1b_processing=="ffsar_bp":
        processing.process()
    elif l1b_processing=="ffsar_omegak":
        if deltat>dt_update_h:
            processing.process(update_h=True)
            previous_burst_time=burst_time[b]
        else:
            processing.process(update_h=False)

    # Getting the focused data
    dict_data=processing.get_data()

    # Writing in the l1b netCDF file
    for k in list_variables:
        file_l1b.variables[k][index_write_nc:index_write_nc+n_multilook]=dict_data[k]

    index_write_nc+=n_multilook

# =============================================================================
# More attributes
# =============================================================================
time_first=date0+timedelta(seconds=float(file_l1b.variables[name_var_time][0]))
time_last=date0+timedelta(seconds=float(file_l1b.variables[name_var_time][-1].data))
file_l1b.first_meas_time="{}-{}-{}T{}:{}:{}Z".format(str(time_first.year).zfill(4),str(time_first.month).zfill(2),str(time_first.day).zfill(2),\
                                                   str(time_first.hour).zfill(2),str(time_first.minute).zfill(2),str(time_first.second).zfill(2))
file_l1b.last_meas_time="{}-{}-{}T{}:{}:{}Z".format(str(time_last.year).zfill(4),str(time_last.month).zfill(2),str(time_last.day).zfill(2),\
                                                   str(time_last.hour).zfill(2),str(time_last.minute).zfill(2),str(time_last.second).zfill(2))
file_l1b.first_meas_lat=float(file_l1b.variables[name_var_lat][0])
file_l1b.last_meas_lat=float(file_l1b.variables[name_var_lat][-1])
file_l1b.first_meas_lon=float(file_l1b.variables[name_var_lon][0])
file_l1b.last_meas_lon=float(file_l1b.variables[name_var_lon][-1])
file_l1b.cycle_number=cycle_number_main
file_l1b.pass_number=pass_number_main


# Geographical selection
file_l1b.shapefile_path=shapefile_path
file_l1b.bbox_lat_min=bbox_lat_min
file_l1b.bbox_lat_max=bbox_lat_max
file_l1b.bbox_lon_min=bbox_lon_min
file_l1b.bbox_lon_max=bbox_lon_max
file_l1b.surface_type=selected_surf_type

# Configuration parameters
file_l1b.l1b_processing=l1b_processing
file_l1b.illumination_time=illumination_time
file_l1b.posting_rate=posting_rate
file_l1b.range_ext_factor=range_ext_factor
file_l1b.zp=zp
file_l1b.hamming_az=int(hamming_az)
file_l1b.hamming_range=int(hamming_range)
    
# =============================================================================
# Closing netCDF files
# =============================================================================
logging.info("L1b processing done. Closing L1a and L1b files.")
file_l1a.close()
file_l1b.close()
