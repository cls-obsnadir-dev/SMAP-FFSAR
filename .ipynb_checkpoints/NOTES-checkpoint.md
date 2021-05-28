# Notes on the use of SMAP

SMAP allows processing S3 l1a files using the back-projection or the omega-k FFSAR processing approaches.

## Geographical selection

Geographical selection of the regions where the data has to be processed can be done with three different ways:
- with a bounding box (lon/lat minimum and maximum values)
- with surface type, based on l1a surface flags
- with a shapefile, containing polygons

The shapefile must contains polygons with an integer attribute `pass_nb`. A given polygon is used for geographical selection only if the relative pass number of the input l1a file is equal to the `pass_nb` value.

It is also possible to specify starting and ending times when launching smap (-s and -e options of launch_smap.py)

## FFSAR processing

By focalization on equidistant along-track surface points with a chosen distance between them, the FFSAR processing allows us to reach some extremely fine resolutions not achievable with UFSAR processing. 

To explain the processing it relies on a "data block" structure, that is shifted along the input file to process it sequentially. 
The size (in number of bursts) of the data block is defined by `n_burst_block`. The synthetic aperture of the FF-SAR processing is performed over the whole data block, so the illumination time of the processing depends on its size. 
The maximum illumination time varies depending on the echoes centering, but, in most cases, 180 bursts are sufficient to catch all the available information (full resolution). The processing of each data block yields `n_sl_output` single-looks that are centered with respect to the data block center and equally spaced in time by the Pulse Repetition Interval (PRI). 
The single-looks are spatially averaged (multi-looked) to obtain the multi-looks (each data block yields `n_multilook` multi-looks, so `n_multilook` needs to be a divisor of `n_sl_output`). 
The sampling frequency of the output product (that contains only the multilooks) is `1 / (n_sl_output / n_multilook * PRI)` in Hz, hence the maximum sampling frequency acheivable (with `n_sl_output=n_multilook`) is equal to 17825Hz. 

<img src = "./img/ffsar_block_burst.PNG">

For non-expert users, the values set by default in the processing to get the optimal FFSAR for S3 are:
* `n_burst_block`=180
* `n_burst_shift`=4
* `n_sl_output`=900
The only parameter chosen by users to define on-ground resolution is the sampling frequency (related to the number of multilooking).

Below a table of correspondance between the multi-looks number and posting rate to process the wanted sampling frequency.

| Posting Rate  | Block shift (`n_burst_shift`)  | Single-looks (`n_sl_output`)  | Multi-looks (`n_multilook`)   |
| ------------- |:------------------------------:|:-----------------------------:|:-----------------------------:|
| 20 Hz         | 4                              |    907                        |    1                          |
| 80 Hz         | 4                              |    900                        |    4                          |
| 100 Hz        | 4                              |    900                        |    5                          |
| 200 Hz        | 4                              |    900                        |    10                         |
| 17825 Hz      | 4                              |    907                        |    907                        |

As mentioned below, the number of multi-looks must be a divisor of the number of single-looks. For transponder analysis, we want to exploit the full-resolution of FFSAR `n_multilook=n_sl_output=907`. For hydrological target it’s different a multi-looking is necessary to reduce waveform noise, since 907 is a prime number we take instead `n_sl_output=896` where `n_multilook=32` is a divisor of 896 and corresponds to 640hz.

After performing the processing on a data block, it is shifted by `n_burst_shift` bursts and operations are repeated. To get non-overlapping multilooks from one block to the following, `n_sl_output` must be smaller or equal to `n_burst_shift * BRI / PRI`.

<img src = "./img/ffsar_block_shift.PNG">

To prevent aliasing artefacts, the processing can be done with an extended window (using the `range_ext_factor` parameter), at the cost of increased computation time.

## Configuration file entries

### Section FILES

- cal2_file : absolute path to .txt file containing 128 floating point values representing gain profile of the LPF (CAL2)
- shapefile : absolute path to .shp shapefile. The shapefile should be structured like the shapefile provided in shp/shapefile_smap.shp. Polygons should have a pass_nb attribute corresponding to the relative pass number that intersect the polygon.

### Section DIR

- output_dir : absolute path to the directory that will contain all the outputs of SMAP (L1b and L2 .nc files, log files, command files)

### Section GENERAL_PARAMETERS

- lat_min, lat_max, lon_min, lon_max : bounding box parameters for geographical selection. Latitude between -90 and 90, longitude between 0 and 360. Can be left blank if not bounding box is to be used.
- surf_type : a combination of 0,1,2 and 3 (separated with commas without spaces) to perform geographical selection based on the surface type (open_ocean_or_semi-enclosed_seas : 0 / enclosed_seas_or_lakes : 1 / continental_ice : 2 / land : 3)

### Section PROCESSING

- l1b_processing : ffsar_bp or ffsar_omegak
- l2_processing : name of L2 processing (retracker) to be applied to the waveforms. Can be stacked (separated with commas, without spaces):OCOG_SAR,PTR,MultiPTR. OCOG_SAR is the standard threshold retracker, PTR fits the waveform with the range PTR (least-square criterion) and MultiPTR performs the PTR retracker with multiple initialisations to catch multiple peaks

### Section PROCESSING_OPTIONS

- n_burst_data_buffer : number of bursts in the data buffer (the whole L1a file is not loaded in memory, only a portion of `n_burst_data_buffer` bursts) (integer >= 1)
- n_burst_block : number of bursts in the data block (integer >= 1)
- n_burst_shift : number of burst to shift the data block (integer >= 1)
- zp : oversampling factor in range (1 or 2)
- hamming_range : whether to apply Hamming weighting in range (yes/no)
- hamming_az : whether to apply Hamming weighting in azimuth, along the burst (yes/no)
- n_fft_az : number of points of azimuth FFT (optimally equal to the closest integer to `BRI/PRI*n_burst_block` to take all the pulses of data block), only for `ffsar_omegak`.
- n_sl_output : number of single-looks generated by data block (integer >= 1)
- n_multilook : number of multi-looks generated by data block (divisor of `n_sl_output`)
- range_ext_factor : extension factor in range (processing is done on `128 * range_ext_factor` range gates)
- dt_update_h : refreshing time for the transfer function, only for `ffsar_omegak` processing (smaller than `BRI * n_burst_shift` for updating the transfer function for each data block), for `ffsar_omegak` only.
- OCOG_threshold_sar : threshold for the OCOG retracker
- MultiPTR_n_estimates : number of initialisations of the Multi PTR retracker (integer >= 1)

## Other comments

- The tracker and range values in the output product are corrected for USO drift and internal path delay. The COG distance is also included.
- The surface height is computed as alt_ffsar - range_ffsar (range_ffsar depends on the retracker that is used). It doesn't include any geophysical correction on the range. The user can find them in L2 land PDGS products. The height is relative to the reference ellipsoid WGS84 (not the geoid).