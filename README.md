# SMAP (Standalone Multi-mission Altimetry Processor)

## Description

SMAP is a standalone altimeter data processor written in Python 3 (3.7.3). It implements in particular the fully-focused SAR (FF-SAR) processing (time-domain back-projection algorithm).
SMAP is currently able to process Sentinel-3 L1a PDGS products.

### Input

1. Sentinel-3 L1A netCDF4 products,
2. a .cfg configuration file,
3. a LTM CAL2 ascii file,
4. optionally, a shapefile to restrict the processing to specfifed areas.

### Output

1. L1b and L2 auto-descriptive netCDF4 files,
2. Command files to launch l1b and l2 processing,
3. log files.

### Directories 

1. `doc/` : documentation of the SMAP,
2. `config/` : configuration files of the SMAP with the options for l1b and l2 processing,
3. `auxi/` : LTM CAL2 ascii file,
4. `shp/` : shapefiles to select some regions of the globe to process (to be indicated in the configuration file),
5. `output/` : output files.

## Prerequisites

- Users are required to install a distribution of Python (we suggest Anaconda,  https://www.anaconda.com/products/individual) and create an environment including Python 3.7.3. More recent versions will not correctly work. 

- Once the installation of Anaconda is completed, the Anaconda prompt shall be launched to create a specific environment (e.g. named "smap_official"):

	conda create -n smap_official python=3.7.3

- The environment shall then be activated with:  

	conda activate smap_official

- In the activated environment, please install the following python packages (use "pip install" for the best compatibility and install the reported version of the packages to replicate the results included in the provided TDS):

On Windows:

	1. pip install numpy==1.17.1
	2. pip install scipy==1.3.1
	3. pip install netcdf4==1.5.1.2
	4. pip install pyproj==2.2.1
	5. pip install pyshp==2.1.0
	6. pip install tqdm==4.35.0
	7. (optionally) pip install numexpr, for faster array computations

On Linux:

	1. $ pip3 install numpy==1.17.1
	2. $ pip3 install scipy==1.3.1
	3. $ pip3 install netcdf4==1.5.1.2
	4. $ pip3 install pyproj==2.2.1
	5. $ pip3 install pyshp==2.1.0
	5. $ pip3 install tqdm==4.35.0
	7. (optionally) pip3 install numexpr, for faster array computations


## Run processing

Before starting to run the processing, it is recommended to the users to read the documentation file `Notes.pdf` in the sub-directory `doc`. In the documentation file, it is explained to users how to edit config files and modify the system paths to according their own configuration (paths to CAL2, shape file, output directory etc.).

### Get starting

To process sequentially all L1a products of a specific folder with SMAP, point the directory of the processor and do:

    python launch_smap.py -c config/config.cfg

given a configuration file `config/config.cfg` in which the input folder path is indicated in the field `input_dir`. In this input folder the users can put all the Sentinel-3 level 1a netCDF4 products wanted (specific date, track, cycle number, etc...).

SMAP creates the output directory specified in the configuration file (`output_dir`) and four subdirectories:

- `l1b/` : L1b netCDF files,
- `l2/` : L2 netCDF files,
- `log/` : log files,
- `cmd/` : command files (bash scripts launching l1b and l2 programs).

### Specific options

- `-c` : to specify the path to the configuration file, as an example:
```
python launch_smap.py -c config/config.cfg
```
will launch all the L1a files of input directory indicated in the configuration file (`input_dir`)

- `-s` or `--start`: starting time to process (format of YYYYMMDDTHHMMSS), as an example:
```
python launch_smap.py -c config/config.cfg -s 20210527T000000 
```
will launch all bursts of the L1a files with time after 2021/05/27 at 00:00:00.

- `-e` or `--end`: ending time to process (format of YYYYMMDDTHHMMSS), as an example:
```
python launch_smap.py -c config/config.cfg -e 20210527T010000 
```
will launch all bursts of the L1a files with time before 2021/05/27 at 01:00:00.

- `-cluster` : creates a PBS job array (with a walltime of 3hours and a memory of 4000mb allocated for each file), as an example:
```
python launch_smap.py -c config/config.cfg -cluster
```
will modify the header of the bash command files in the subdirectory `cmd` of the output directory as:
```bash
#!/bin/bash
#PBS -N SMAP
#PBS -J 0-#l1a_file
#PBS -l select=1:ncpus=1:mem=4000mb
#PBS -l walltime=03:00:00
```
where `#l1a_file` is replaced by the number of L1a files and launch them.

- `-dryrun` : creates bash command files only, as an example:
```
python launch_smap.py -c config/config.cfg -dryrun
```
will create the bash command file(s) without launching them. To launch them:
```
qsub output_dir/cmd/*.sh
```
where `output_dir` is the output directory.

All the options can be combined.

### Test Data Set (TDS)

Test data set (TDS) output, launched by CLS with SMAP code, is provided in the sub-directory `output` as a reference. Information on how to launch and visualize the TDS can be found in the documentation file `Notes.docx` in the sub-directory `doc`.

## References

- [Egido, A., & Smith, W. H. (2016). Fully focused SAR altimetry: theory and applications. IEEE Transactions on Geoscience and Remote Sensing, 55(1), 392-406.] (https://ieeexplore.ieee.org/abstract/document/7579570)
- [Guccione, P.; Scagliola, M.; Giudici, D. 2D Frequency Domain Fully Focused SAR Processing for High PRF Radar Altimeters. Remote Sens. 2018, 10, 1943.] 
(https://doi.org/10.3390/rs10121943)

## Authors

- Pierre Rieu - CLS
- Samira Amraoui - CLS
- Marco Restano - SERCO c/o ESA-ESRIN