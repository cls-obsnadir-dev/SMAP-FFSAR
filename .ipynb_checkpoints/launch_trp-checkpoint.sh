#PBS -N trp_s3_ms
#PBS -o /work/ALT/sentinel6/samraoui/ASCII_table/OUTPUT_PBS/FFSAR_BP/
#PBS -e /work/ALT/sentinel6/samraoui/ASCII_table/OUTPUT_PBS/FFSAR_BP/
#PBS -l select=1:ncpus=24:mem=184G
#PBS -l walltime=24:00:00

rep_code="/work/ALT/sentinel6/samraoui/smap_delivery_review"
rep_input="/work/ALT/sentinel6/samraoui/input/TRP/S3A"
file_input="S3A_SR_1_SRA_A__20180628T192512_20180628T201541_20180723T214218_3029_033_013______LN3_O_NT_003.SEN3/measurement_l1a.nc"

python $rep_code/launch_smap.py $rep_input/$file_input -c $rep_code/config_trp.cfg