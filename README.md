## mpi4py based feature extraction/patch generation for deep learning applications of CT-images

### Highlight

(1) Performs patch based feature extracting from raw training set in Parallel<br>
(2) Includes different options for data normalization<br>
(3) Includes translation and scaling based augmentation<br>
(4) Includes air thresholding to discard non-contrast regions

```
usage: main.py [-h] --input-folder INPUT_FOLDER [--output-fname OUTPUT_FNAME] [--patch-size PATCH_SIZE]
               [--normalization-type NORMALIZATION_TYPE] [--tensor-format TENSOR_FORMAT] [--random_N] [--rot_augment]
               [--ds_augment] [--air_threshold] [--blurr_n_noise] [--mpi_run]

Storing input-target images as patches in h5 format from all patient data / category sets

optional arguments:
  -h, --help            show this help message and exit
  --input-folder        directory name containing input-target images for all categories/patients
  --output-fname        output filename to save patched h5 file
  --patch-size          p96 or p75 or p55 or p42 or p24 or p12. p96 yields 96x96 patched window
  --normalization-type  unity_independent or unity_wrt_ld or std_independent or std_wrt_ld or None, for more info
                        look at the function img_pair_normalization in utils.py
  --tensor-format       other option is tf. Depending upon the DL API tool, h5 input and target patches are saved
                        accordingly. Eg torch tensor [batch_size, c, h, w]
  --random_N            extracts random N complimentary images from input - target folders. For more info refer to
                        in-built options
  --rot_augment         employs rotation based augmentation
  --ds_augment          incorporates downscale based data augmentation
  --air_threshold       removes patches devoid of contrast
  --blurr_n_noise       whether or not you want to add noise and blurr input data. This option is only implemented
                        for serial implementation and not for mpi for now
  --mpi_run             if you want to employ mpi based patch synthesis
```

### Example usage

`$ mpiexec -n 8 python main.py --input-folder './raw_data' --output-fname './results/training_patches.h5' --patch-size 'p55' --mpi_run --rot_augment --ds_augment --air_threshold`<br>
or<br>
$ chmod +x gen_patches.sh<br>
$ ./gen_patches

### Results
<img src="/sanity_check/raw_data/norm_None_patch_size_p55/hr_input_sub_img_rand_5753.png" alt="Target patch fig"/>
<img src="/sanity_check/raw_data/norm_None_patch_size_p55/lr_input_sub_img_rand_5753.png" alt="Input patch fig"/>

### Package requirements

install mpicc compiler as conda install -c anaconda mpi4py<br>
use pip to install numpy, h5py, matplotlib, pydicom, glob, random, opencv-python

### References
- Some demo images in the raw_data folder are imported from the Low Dose CT Grand Challenge<br>
https://www.aapm.org/GrandChallenge/LowDoseCT/
- L. Dalcin, P. Kler, R. Paz, and A. Cosimo, Parallel Distributed Computing using Python, Advances in Water Resources, 34(9) 1124-1139, 2011. http://dx.doi.org/10.1016/j.advwatres.2011.04.013
- L. Dalcin, R. Paz, M. Storti, and J. D’Elia, MPI for Python: performance improvements and MPI-2 extensions, Journal of Parallel and Distributed Computing, 68(5):655-662, 2008. http://dx.doi.org/10.1016/j.jpdc.2007.09.005
- L. Dalcin, R. Paz, and M. Storti, MPI for Python, Journal of Parallel and Distributed Computing, 65(9):1108-1115, 2005. http://dx.doi.org/10.1016/j.jpdc.2005.03.010
### Contact
Prabhat KC
prabhat.kc077@gmail.com
