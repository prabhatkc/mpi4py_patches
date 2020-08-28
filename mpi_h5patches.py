import os
import utils
import numpy as np
import sys

import glob_funcs as gf

import argparse
import mpi_utils 
from mpi4py import MPI
import random
import h5py


def makeh5patches_in_mpi(args):
	# MPI declarations
	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	size = comm.Get_size()
	dest = 0
	root = 0

	# -------------------------------------
	# Variables to be used for 
	# 	 Reduce Operations
	# -----------------------------------
	partition_img_sum = np.zeros(1, dtype='d')
	# float declarations
	global_pre_norm_tar_min  = np.zeros(1, dtype='float32')
	global_pre_norm_tar_max  = np.zeros(1, dtype='float32')
	global_post_norm_tar_min = np.zeros(1, dtype='float32')
	global_post_norm_tar_max = np.zeros(1, dtype='float32')
	# dict diclarations
	partitioned_data = None

	# ------------------------------------
	# Read Raw images &
	# Declare required variables
	# to be broadcasted
	# -----------------------------------
	if rank==root:

		print('\n----------------------------------------')
		print('Command line arguements')
		print('----------------------------------------')
		for i in args.__dict__: print((i),':',args.__dict__[i])
		print('----------------------------------------')
		
		# read all images in the training dataset
		all_input_paths, all_target_paths = mpi_utils.img_paths4rm_training_directory(args)

		# No. of images (or image chunck) to be processed by each processor
		# Ensuring that N_images % nproc = 0
		nproc = size
		if (np.mod(len(all_target_paths), nproc)!=0):
			print('\n==>Removing last', np.mod(len(all_target_paths), nproc), 'image path(s)', end =' ')
			print('from the training dataset such that the mod(N_images,nproc)=0')
			N_last_paths = np.mod(len(all_target_paths), nproc)
			all_target_paths = all_target_paths[:-N_last_paths]
			all_input_paths  = all_input_paths [:-N_last_paths]

		chunck = int(len(all_target_paths)/nproc) 
		print('  ',len(all_target_paths),'LR-HR image pairs are processed by', nproc, \
			'processors; with each rank processing', chunck, 'img-pairs')
		'''
		print('\nTraining input image paths:')
		print(all_input_paths)
		print('\n\nTraining target image paths:')
		print(all_target_paths)
		'''
		# input-output variables from each rank
		bcasted_input_data = {'all_target_paths': all_target_paths, 'all_input_paths': all_input_paths,\
							  'chunck': chunck, 'nproc':nproc}
	else:
		bcasted_input_data  = None

	bcasted_input_data  = comm.bcast(bcasted_input_data, root=root)

	partitioned_data = dict(mpi_utils.partition_read_normalize_n_augment(args, bcasted_input_data, rank))
	comm.Barrier()

	comm.Reduce(partitioned_data['chunck_pre_norm_min'],  global_pre_norm_tar_min,  op=MPI.MIN, root=dest)
	comm.Reduce(partitioned_data['chunck_pre_norm_max'],  global_pre_norm_tar_max,  op=MPI.MAX, root=dest)
	comm.Reduce(partitioned_data['chunck_post_norm_min'], global_post_norm_tar_min, op=MPI.MIN, root=dest)
	comm.Reduce(partitioned_data['chunck_post_norm_max'], global_post_norm_tar_max, op=MPI.MAX, root=dest)
	comm.Barrier()

	# pixel count from each rank collected and appended to an array
	arr_of_InputCounts_in_ranks = np.array(comm.gather(len(partitioned_data['chunck_input_patch_inbuff']), dest))
	arr_of_TargetCounts_in_ranks = np.array(comm.gather(len(partitioned_data['chunck_target_patch_inbuff']), dest))

	# Allocate arrays to collect all the processed patches from all the ranks
	if (rank == dest): 
		all_input_patch_buff = np.zeros(sum(arr_of_InputCounts_in_ranks), dtype='float32')
		all_target_patch_buff = np.zeros(sum(arr_of_TargetCounts_in_ranks), dtype='float32')
	else: 
		all_input_patch_buff = None; all_target_patch_buff=None

	# Collect all the processed (input-target) patches from all ranks in buffers
	comm.Gatherv(sendbuf=partitioned_data['chunck_input_patch_inbuff'], \
		recvbuf=(all_input_patch_buff, arr_of_InputCounts_in_ranks), root=dest)
	comm.Gatherv(sendbuf=partitioned_data['chunck_target_patch_inbuff'], \
		recvbuf=(all_target_patch_buff, arr_of_TargetCounts_in_ranks), root=dest)
	comm.Barrier()

	#-------------------------------------
	#  final patch formating 
	#       as per 
	#  command line args
	#--------------------------------------
	if (rank==dest):
		# changing the patch sizes of input patches if bicubic initialization is True
		iph, ipw = args.input_size, args.input_size
		
		sub_input_of_all_inputs = mpi_utils.bufftoarr(all_input_patch_buff, sum(arr_of_InputCounts_in_ranks), iph, ipw, args.channel)
		sub_label_of_all_labels = mpi_utils.bufftoarr(all_target_patch_buff, sum(arr_of_TargetCounts_in_ranks), args.label_size, args.label_size, args.channel)
		
		# --------------------------
		# Shuffling the patches     
		# --------------------------
		""" for now shuffling is commented out as shuffling can easily be implemented 
		at the time of run using pytorch or tf based data sampler. 
		Also, current shuffling code is more of a bottle neck and needs to be implemented
		in parallel with array based shuffling rather than using index shuffle. 
		if args.shuffle_patches:
			Npatches = len(sub_input_of_all_inputs)
			shuffled_Npatches_arr = np.arange(Npatches)
			np.random.shuffle(shuffled_Npatches_arr)
			sub_input_of_all_inputs = sub_input_of_all_inputs[shuffled_Npatches_arr, :, :, :]
			sub_label_of_all_labels = sub_label_of_all_labels[shuffled_Npatches_arr, :, :, :]
		"""

		# -----------------------------------------------------
		# Sanity check
		# making patch plot of random patches for sanity check
		#------------------------------------------------------
		if args.sanity_plot_check:
			# declaring path to save sanity check results
			sanity_chk_path = 'sanity_check/'+((args.input_folder).split('/'))[-1] + '/norm_' + str(args.normalization_type) + '_patch_size_' + str(args.patch_size)
			if not os.path.isdir(sanity_chk_path): os.makedirs(sanity_chk_path)
			window = 12
			lr_N = len(sub_input_of_all_inputs)
			rand_num=random.sample(range(lr_N-window), 5)
			for k in range(len(rand_num)):
				s_ind  = rand_num[k]
				e_ind  =  s_ind+window
				lr_out_path = os.path.join(sanity_chk_path+'/lr_input_sub_img_rand_'+str(rand_num[k])+'.png')
				hr_out_path = os.path.join(sanity_chk_path+'/hr_input_sub_img_rand_'+str(rand_num[k])+'.png')
				gf.multi2dplots(3, 4, sub_input_of_all_inputs[s_ind:e_ind, :, :, 0], 0, passed_fig_att = {"colorbar": False, "figsize":[4,4], "out_path": lr_out_path})
				gf.multi2dplots(3, 4, sub_label_of_all_labels[s_ind:e_ind, :, :, 0], 0, passed_fig_att = {"colorbar": False, "figsize":[4,4], "out_path": hr_out_path})
		
		# -----------------------------------------------------------------	
		# Tensor format
		# torch reads tensor as [batch_size, channels, height, width]
		# tf reads tensor as [batch_size, height, width, batch_size]
		# -----------------------------------------------------------------	
		if args.tensor_format == 'torch':
			sub_input_of_all_inputs = np.transpose(sub_input_of_all_inputs, (0, 3, 1, 2))
			sub_label_of_all_labels = np.transpose(sub_label_of_all_labels, (0, 3, 1, 2))
		elif args.tensor_format == 'tf':
			sub_input_of_all_inputs = sub_input_of_all_inputs
			sub_label_of_all_labels = sub_label_of_all_labels
		print("\n==>Shape of the overall input  subimages: {}".format(sub_input_of_all_inputs.shape))
		print("   Shape of the overall target subimages: {}".format(sub_label_of_all_labels.shape))
		print("   Range of Target Image stack changes from (%.4f, %.4f) to (%.4f, %.4f)\n   due to " %\
		(global_pre_norm_tar_min, global_pre_norm_tar_max,\
		global_post_norm_tar_min, global_post_norm_tar_max), args.normalization_type)
		#print('final sum of input, target is:', np.sum(sub_input_of_all_inputs), np.sum(sub_label_of_all_labels))

		# --------------------
		# creating h5 file
		#---------------------
		output_folder = os.path.split(args.output_fname)[0]
		if not os.path.isdir(output_folder): os.makedirs(output_folder)
		hf = h5py.File(args.output_fname, mode='w')
		hf.create_dataset('input', data=sub_input_of_all_inputs)
		hf.create_dataset('target', data=sub_label_of_all_labels)
		hf.close()

	comm.Barrier()