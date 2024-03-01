<< COMMENT
#------------------------------------------------------------------------------------------------------------------#
#                   WARNINGS
#------------------------------------------------------------------------------------------------------------------#
(1) If your training data size is too big i.e. in the order of 10s or 100s of GB, 
    you may incur segmentation faults as the data array size might become larger than what your machine/processor 
    can process at a given time. 
    So it is advised that you split your training raw data as Train_data1, Train_data2, 
    Train_data3 or so on such that each Train_data3 is less than 5 GB or your machine's each processor's threshold/memory 
    space.
    
(2) If you opt for normalization other than "None", do not use out-dtype such as int16 or uint16. 

COMMENT
cmd_opt=$1
cmd_opt1="seriel_run"
cmd_opt2="mpi_run"
cmd_opt3="mixed_train_data"
cmd_opt4="sr_mpi_run"

# ------------------------------------
# seriel_run
# ------------------------------------
if [[ "$cmd_opt" == "$cmd_opt1" ]]
then
  OUTPUT_FNAME='./serial_patches/p55_ldhd_patches.h5'
  python main.py --input-folder 'raw_data' --output-fname $OUTPUT_FNAME --patch-size 'p55' --air_threshold --out-dtype 'float32' \
  --sanity_plot_check --air_threshold --ds_augment --rot_augment --dose_blend --nsplit 2 --multi-patients

# ----------------------------------------------------------
# mpi run
# here mpiexec -n 4 means that the code is going to use 4 processors  
# ----------------------------------------------------------
elif [[ "$cmd_opt" == "$cmd_opt2" ]]
then
  # direct patching without any shuffling
  OUTPUT_FNAME='./mpi_patches/p55_ldhd_patches.h5'
  mpiexec -n 4 \
  python main.py --input-folder 'raw_data' --output-fname $OUTPUT_FNAME --patch-size 'p55' --out-dtype 'float16' \
  --air_threshold --ds_augment --rot_augment --mpi_run --sanity_plot_check --dose_blend --nsplit 2 \
  --input-gen-folder 'quarter_3mm_sharp_sorted' --target-gen-folder 'full_3mm_sharp_sorted' --multi-patients
  # shuffle example 
  # OUTPUT_FNAME='./mpi_patches/p96_val_patches.h5'
  # mpiexec -n 4 python main.py --input-folder 'raw_data' --output-fname $OUTPUT_FNAME --patch-size 'p96' --out-dtype 'float16' \
  # --air_threshold --ds_augment --rot_augment --mpi_run --sanity_plot_check --dose_blend --nsplit 2 \
  # --input-gen-folder 'quarter_3mm_sharp_sorted' --target-gen-folder 'full_3mm_sharp_sorted' --shuffle-patches 'torch_shuffle'

# ----------------------------------------------------------
# for mixed training (i.e. mixing acquisition of patients)
# following executes shuffling scans at patient level
# ----------------------------------------------------------
elif [[ "$cmd_opt" == "$cmd_opt3" ]]
then
  # remove dummy place holder files from previous executions (if any)
  rm -r raw_data_mixed_cp
  rm -r pre_randomize
  rm -r placeholder
  python pre_patch_scan_shuffle.py --input-folder 'raw_data_mixed' --output-folder 'pre_randomize' --nsplit 2 --multi-patients

  # ----------------------------------------------------------
  # following executes shuffling scans at patient level
  # ----------------------------------------------------------
  OUTPUT_FNAME='./mpi_patches/part_0_mx_shuffled.h5'
  mpiexec -n 1 \
  python main.py --input-folder 'pre_randomize/part_0/mx_data' --output-fname $OUTPUT_FNAME --patch-size 'p55' --out-dtype 'float16' \
  --air_threshold --ds_augment --rot_augment --mpi_run --sanity_plot_check --dose_blend --nsplit 2 \
  --input-gen-folder 'quarter_3mm' --target-gen-folder 'full_3mm' --shuffle-patches 'np_shuffle'

elif [[ "$cmd_opt" == "$cmd_opt4" ]]
then 
  OUTPUT_FNAME='./sr_mpi_patches/p96_sr_patches.h5' 
  mpiexec -n 4 
  python main_sr.py --input-folder 'raw_data' --output-fname ${OUTPUT_FNAME} --patch-size 'p96' --out-dtype 'float16' \
  --air_threshold --sanity_plot_check --nsplit 2 --input-gen-folder 'full_3mm_sharp_sorted' --target-gen-folder 'full_3mm_sharp_sorted' \
  --shuffle-patches None --scale 4 --img-format 'dicom' --normalization-type 'unity_independent' --mpi_run --rot_augment --ds_augment\
  --multi-patients

else
  echo ""
  echo "re-check cmd line option in the demo_run.sh"
  echo ""
fi