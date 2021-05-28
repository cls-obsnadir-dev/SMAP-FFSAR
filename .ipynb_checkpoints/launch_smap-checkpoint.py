#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import configparser
import sys
import subprocess
import netCDF4
import platform

# =============================================================================
# Platform
# =============================================================================

os_system=platform.system()
if os_system not in ["Linux","Windows","Darwin"]:
    sys.stderr.write("Error: OS not supported.\n")
    sys.exit(1)

# =============================================================================
# Parsing arguments
# =============================================================================

parser=argparse.ArgumentParser()
parser.add_argument("l1a_infile",help="l1a files to process",nargs="+")
parser.add_argument("-c","--config_file",help="path to .cfg config file")
parser.add_argument("-cluster",help="creates a PBS job array",action='store_true')
parser.add_argument("-dryrun",help="creates bash command files only",action="store_true")
parser.add_argument("-s","--start",help="date for start of processing YYYYMMDDTHHMMSS")
parser.add_argument("-e","--end",help="date for end of processing YYYYMMDDTHHMMSS")
args = parser.parse_args()

# =============================================================================
# Config file
# =============================================================================

current_file=__file__
current_dir=os.path.dirname(os.path.realpath(current_file))

if args.config_file is None:
    config_filename=os.path.join(current_dir,"config.cfg")
else:
    config_filename=os.path.realpath(args.config_file)

config=configparser.ConfigParser()
try:
    config.read(config_filename)
except:
    sys.stderr.write("Error: unable to open/read config file.\n")
    sys.exit(1)
    
# =============================================================================
# Executables
# =============================================================================

l1b_ex=os.path.join(current_dir,"main_l1b.py")
l2_ex=os.path.join(current_dir,"main_l2.py")

# =============================================================================
# Creating directories
# =============================================================================

dir_out=os.path.realpath(config.get('DIR','output_dir'))
dir_out_l1b=os.path.join(dir_out,"l1b")
dir_out_l2=os.path.join(dir_out,"l2")
dir_out_log=os.path.join(dir_out,"log")
dir_out_cmd=os.path.join(dir_out,"cmd")
os.makedirs(dir_out_l1b,exist_ok=True)
os.makedirs(dir_out_l2,exist_ok=True)
os.makedirs(dir_out_log,exist_ok=True)
os.makedirs(dir_out_cmd,exist_ok=True)

# =============================================================================
# Removing existing command files
# =============================================================================

# for file_cmd in os.listdir(dir_out_cmd):
#     os.remove(os.path.join(dir_out_cmd,file_cmd))

# =============================================================================
# Loop over input files
# =============================================================================

n_file_to_process=len(args.l1a_infile)
list_bash_cmd_filename=[]

for filename_l1a in args.l1a_infile:

    filename_l1a_real=os.path.realpath(filename_l1a)
    
    # =============================================================================
    # Opening the l1a file to get the basename    
    # =============================================================================
    
    try: 
        file_l1a=netCDF4.Dataset(filename_l1a_real,"r")
        basename=file_l1a.product_name.split(".")[0]
        file_l1a.close()
    except:
        sys.stderr.write("Unable to open file {}. Skipping it.\n".format(filename_l1a_real))
        continue
    
    # =============================================================================
    # Output files    
    # =============================================================================
        
    filename_l1b=os.path.join(dir_out_l1b,basename+"_l1b.nc")
    filename_l1b_log=os.path.join(dir_out_log,basename+"_l1b.log")
    filename_l2=os.path.join(dir_out_l2,basename+"_l2.nc")
    filename_l2_log=os.path.join(dir_out_log,basename+"_l2.log")
    
    # =============================================================================
    # Writing the command .sh file, with calls to the main_l1b.py and main_l2.py programs.
    # If the cluster option is activated, the input file is copied to ${TMPDIR}
    # and the output files are created inside ${TMPDIR}. At the end of the processing, the outputs 
    # are copied to the specified directory.
    # =============================================================================
        
    if os_system=="Windows":
        bash_cmd_filename=os.path.join(dir_out_cmd,basename+"_cmd.cmd")
    elif os_system=="Linux" or os_system=="Darwin":
        bash_cmd_filename=os.path.join(dir_out_cmd,basename+"_cmd.sh")
    list_bash_cmd_filename.append(bash_cmd_filename)
    
    bash_cmd_file=open(bash_cmd_filename,"w")
    bash_cmd_file.write("#!/bin/bash\n\n")

    if args.cluster:
        
        bash_cmd_file.write("cp {} ".format(filename_l1a_real) +"${TMPDIR}"+"/{}_l1a.nc\n\n".format(basename))
        
        if args.start is not None and args.end is not None:
        
            bash_cmd_file.write(l1b_ex +
                                " ${TMPDIR}"+"/{}_l1a.nc".format(basename) +
                                " -c {}".format(config_filename) +
                                " -o ${TMPDIR}"+"/{}_l1b.nc".format(basename) +
                                " -l ${TMPDIR}"+"/{}_l1b.log".format(basename) +
                                " -s {}".format(args.start) +
                                " -e {}\n\n".format(args.end))
            
        elif args.start is not None:
            
            bash_cmd_file.write(l1b_ex +
                                " ${TMPDIR}"+"/{}_l1a.nc".format(basename) +
                                " -c {}".format(config_filename) +
                                " -o ${TMPDIR}"+"/{}_l1b.nc".format(basename) +
                                " -l ${TMPDIR}"+"/{}_l1b.log".format(basename) +
                                " -s {}\n\n".format(args.start))
            
        elif args.start is not None:
            
            bash_cmd_file.write(l1b_ex +
                                " ${TMPDIR}"+"/{}_l1a.nc".format(basename) +
                                " -c {}".format(config_filename) +
                                " -o ${TMPDIR}"+"/{}_l1b.nc".format(basename) +
                                " -l ${TMPDIR}"+"/{}_l1b.log".format(basename) +
                                " -e {}\n\n".format(args.end))
            
        else:
            
            bash_cmd_file.write(l1b_ex +
                                " ${TMPDIR}"+"/{}_l1a.nc".format(basename) +
                                " -c {}".format(config_filename) +
                                " -o ${TMPDIR}"+"/{}_l1b.nc".format(basename) +
                                " -l ${TMPDIR}"+"/{}_l1b.log\n\n".format(basename))

        bash_cmd_file.write("cp ${TMPDIR}"+"/{}_l1b.log {}\n\n".format(basename,filename_l1b_log))

        bash_cmd_file.write("if [ -e ${TMPDIR}"+"/{}_l1b.nc".format(basename)+" ] \n")
        bash_cmd_file.write("then\n")
        
        bash_cmd_file.write("cp ${TMPDIR}"+"/{}_l1b.nc {}\n".format(basename,filename_l1b))
        
        bash_cmd_file.write(l2_ex +
                            " ${TMPDIR}"+"/{}_l1b.nc".format(basename) +
                            " -c {}".format(config_filename) +
                            " -o ${TMPDIR}"+"/{}_l2.nc".format(basename) +
                            " -l ${TMPDIR}"+"/{}_l2.log\n\n".format(basename))
        
        bash_cmd_file.write("cp ${TMPDIR}"+"/{}_l2.nc {}\n".format(basename,filename_l2))
        bash_cmd_file.write("cp ${TMPDIR}"+"/{}_l2.log {}\n\n".format(basename,filename_l2_log))

        bash_cmd_file.write("fi\n")
        
        bash_cmd_file.write("rm ${TMPDIR}"+"/{}_l1a.nc\n".format(basename))
        bash_cmd_file.write("rm ${TMPDIR}"+"/{}_l1b.log\n".format(basename))

        bash_cmd_file.write("if [ -e ${TMPDIR}"+"/{}_l1b.nc".format(basename)+" ] \n")
        bash_cmd_file.write("then\n")
        bash_cmd_file.write("rm ${TMPDIR}"+"/{}_l1b.nc\n".format(basename))
        bash_cmd_file.write("rm ${TMPDIR}"+"/{}_l2.nc\n".format(basename))
        bash_cmd_file.write("rm ${TMPDIR}"+"/{}_l2.log\n\n".format(basename))
        bash_cmd_file.write("fi\n")
        
    else:
        
        if os_system=="Linux" or os_system=="Darwin":
            
            if args.start is not None and args.end is not None:
                bash_cmd_file.write("python {} {} -c {} -o {} -l {} -s {} -e {}\n\n".format(l1b_ex,filename_l1a_real,config_filename,filename_l1b,filename_l1b_log,args.start,args.end))
            elif args.start is not None:
                bash_cmd_file.write("python {} {} -c {} -o {} -l {} -s {}\n\n".format(l1b_ex,filename_l1a_real,config_filename,filename_l1b,filename_l1b_log,args.start))
            elif args.end is not None:
                bash_cmd_file.write("python {} {} -c {} -o {} -l {} -e {}\n\n".format(l1b_ex,filename_l1a_real,config_filename,filename_l1b,filename_l1b_log,args.end))
            else:
                bash_cmd_file.write("python {} {} -c {} -o {} -l {}\n\n".format(l1b_ex,filename_l1a_real,config_filename,filename_l1b,filename_l1b_log))
                
            bash_cmd_file.write("if [ -e {} ]\n".format(filename_l1b))
            bash_cmd_file.write("then\n")
            bash_cmd_file.write("python {} {} -c {} -o {} -l {}\n\n".format(l2_ex,filename_l1b,config_filename,filename_l2,filename_l2_log))
            bash_cmd_file.write("fi\n")
        elif os_system=="Windows":
            
            if args.start is not None and args.end is not None:
                bash_cmd_file.write("python {} {} -c {} -o {} -l {} -s {} -e {}\n\n".format(l1b_ex,filename_l1a_real,config_filename,filename_l1b,filename_l1b_log,args.start,args.end))
            elif args.start is not None:
                bash_cmd_file.write("python {} {} -c {} -o {} -l {} -s {}\n\n".format(l1b_ex,filename_l1a_real,config_filename,filename_l1b,filename_l1b_log,args.start))
            elif args.end is not None:
                bash_cmd_file.write("python {} {} -c {} -o {} -l {} -e {}\n\n".format(l1b_ex,filename_l1a_real,config_filename,filename_l1b,filename_l1b_log,args.end))
            else:
                bash_cmd_file.write("python {} {} -c {} -o {} -l {}\n\n".format(l1b_ex,filename_l1a_real,config_filename,filename_l1b,filename_l1b_log))
                
            bash_cmd_file.write("python {} {} -c {} -o {} -l {}\n\n".format(l2_ex,filename_l1b,config_filename,filename_l2,filename_l2_log))

    bash_cmd_file.close()
        
    if os_system=="Linux" or os_system=="Darwin":
        rchmod=subprocess.run(['chmod','700',bash_cmd_filename])
        if rchmod.returncode!=0:
            sys.stderr.write("Error while chmod file {}\n".format(bash_cmd_filename))
    
# =============================================================================
# If the cluster option is activated, a job array is created 
# =============================================================================
if args.cluster:
        
    pbs_filename=os.path.join(dir_out_cmd,"job_array_processing.pbs")
    pbs_filename_out=os.path.join(dir_out_cmd,"job_array_processing.out")
    pbs_filename_err=os.path.join(dir_out_cmd,"job_array_processing.err")
    pbs_file=open(pbs_filename,"w")
        
    pbs_file.write("#!/bin/bash\n\n")
    pbs_file.write("#PBS -N SMAP\n")
    if n_file_to_process>1:
        pbs_file.write("#PBS -J 0-"+str(n_file_to_process)+"\n")
    pbs_file.write("#PBS -l select=1:ncpus=1:mem=4000mb\n")
    pbs_file.write("#PBS -l walltime=03:00:00\n")
    pbs_file.write("#PBS -o {}\n".format(pbs_filename_out))
    pbs_file.write("#PBS -e {}\n".format(pbs_filename_err))
    pbs_file.write("\n")
    if n_file_to_process>1:
        pbs_file.write("file=`ls "+dir_out_cmd+"/S3*.sh | head -n $PBS_ARRAY_INDEX | tail -1`\n\n")
    else:
        pbs_file.write("file=`ls "+dir_out_cmd+"/S3*.sh | head -n 1 | tail -1`\n\n")
    pbs_file.write("${file}\n\n")

    pbs_file.close()
    
    rchmod=subprocess.run(['chmod','700',pbs_filename])
    if rchmod.returncode!=0:
        sys.stderr.write("Error while chmod file {}\n".format(pbs_filename))
        
# =============================================================================
# Launching the processing
# =============================================================================
if not args.dryrun:
    
    # =============================================================================
    # Launching the job array    
    # =============================================================================
    if args.cluster:
        
        sys.stdout.write("Launching job array with qsub.\n")
        qsub=subprocess.run(['qsub',pbs_filename])
        if qsub.returncode!=0:
            sys.stderr.write("Error while launching job array with qsub.\n")
    
    # =============================================================================
    # Launching the .sh command files one by one       
    # =============================================================================
    else:

        for bash_cmd_filename in list_bash_cmd_filename:

            sys.stdout.write("Launching {}\n".format(bash_cmd_filename))
            proc=subprocess.run([bash_cmd_filename],shell=True)
            if proc.returncode!=0:
                sys.stderr.write("Error while launching {}\n".format(bash_cmd_filename))
