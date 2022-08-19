<< COMMENT
#------------------------------------------------------------------------------------------------------------------#
#  									OPTIONS
#------------------------------------------------------------------------------------------------------------------#
usage: main.py [-h] --input-folder INPUT_FOLDER [--output-fname OUTPUT_FNAME] [--patch-size PATCH_SIZE]
               [--normalization-type NORMALIZATION_TYPE] [--tensor-format TENSOR_FORMAT] [--random_N] [--rot_augment]
               [--ds_augment] [--air_threshold] [--blurr_n_noise] [--mpi_run] [--dose_blend] [--sanity_plot_check] 
               [--nsplit NSPLIT] [--out-dtype OUT_DTYPE] [--input_gen_folder INPUT_GEN_FOLDER]
               [--target_gen_folder TARGET_GEN_FOLDER]

Storing input-target images as patches in h5 format from all patient data /
category sets

optional arguments:
  -h, --help            show this help message and exit
  --input-folder        directory name containing images
  --output-fname        output filename to save patched h5 file
  --patch-size          p96 or p75 or p55 or p42 or p24 or p12. p96 yields 96x96 patched window
  --normalization-type  None or unity_independent or unity_wrt_ld or std_independent or std_wrt_ld. 
                        For more info look at function img_pair_normalization in utils.py
  --tensor-format       other option is tf. Depending upon the DL API tool, h5 input and target patches are saved accordingly. 
                        Eg. torch tensor [batch_size, c, h, w]
  --random_N            extracts random N complimentary images from input - target folders. For more info refer to 
                        in-built options.
  --rot_augment         employs rotation-based augmentation
  --ds_augment          incorperate downscale based data augmentation
  --air_threshold       removes patches devoid of contrast
  --blurr_n_noise       whether or not you want to add noise and blurr input data. Non-funtional in for the mpi-run. 
                        Only works in serial run (for now).
  --mpi_run             if you want to employ mpi-based parallel computation
  --dose_blend          if you want to employ dose blend-base data
                        augmendation
  --sanity_plot_check   if you want to view some of the patched plots
  --nsplit              no. of h5 files containing n chunks of patches
  --out-dtype           array type of output h5 file. Options include float32

#------------------------------------------------------------------------------------------------------------------#
#                   WARNINGS
#------------------------------------------------------------------------------------------------------------------#
(1) If your training data size is too big i.e. in the order of10s or 100s of GB, 
    you may incur segmentation faults as the data array size might become larger than what your machine/processor 
    can process at a given time. So it is advised that you split your training raw data as Train_data1, Train_data2, 
    Train_data3 or so on such that each Train_data3 is less than 5 GB or your machine's each processor's threshold/memory 
    space.
    
(2) If you opt for normalization other than "None", do not use out-dtype such as int16 or uint16. 


COMMENT
# seriel run
# OUTPUT_FNAME='./serial_patches/val_patches.h5'
# python main.py --input-folder 'raw_data' --output-fname $OUTPUT_FNAME --patch-size 'p55' --air_threshold --out-dtype 'float32' \
# --sanity_plot_check --air_threshold --ds_augment --rot_augment --dose_blend --nsplit 2

# mpi run
# here mpiexec -n 4 means that the code is going to use 4 processors  
OUTPUT_FNAME='./mpi_patches/val_patches.h5'
mpiexec -n 4 python main.py --input-folder 'raw_data' --output-fname $OUTPUT_FNAME --patch-size 'p55' --out-dtype 'float16' \
--air_threshold --ds_augment --rot_augment --mpi_run --sanity_plot_check --dose_blend --nsplit 2