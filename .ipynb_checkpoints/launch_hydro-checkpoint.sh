#PBS -N hydro_s3_ms
#PBS -o /work/ALT/sentinel6/samraoui/ASCII_table/OUTPUT_PBS/FFSAR_BP/
#PBS -e /work/ALT/sentinel6/samraoui/ASCII_table/OUTPUT_PBS/FFSAR_BP/
#PBS -l select=1:ncpus=24:mem=184G
#PBS -l walltime=24:00:00

rep_code="/work/ALT/sentinel6/samraoui/smap_delivery_review"
rep_input="/work/ALT/sentinel6/samraoui/input/OCEAN/S3A"
file_input="S3A_SR_1_SRA_A__20190730T101715_20190730T110744_20190824T200439_3029_047_279______LN3_O_NT_003.SEN3/measurement_l1a.nc"

python $rep_code/launch_smap.py $rep_input/$file_input -c $rep_code/config_hydro.cfg