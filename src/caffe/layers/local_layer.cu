// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"

#include <iostream>
#include <fstream>

//#define DEBUG_LOCAL_FP

namespace caffe {

	template <typename Dtype>
	Dtype LocallyConnectedLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {

			//BUG FOUND! update memory before copy to slave memory!!!
			CUDA_CHECK(cudaDeviceSynchronize());
			//Caffe::switch_to_master_device();

			const Dtype* bottom_data = bottom[0]->gpu_data();
			Dtype* top_data = (*top)[0]->mutable_gpu_data();
			Dtype* col_data = col_buffer_.mutable_gpu_data();
			
			Dtype* trans_data = trans_buffer_.mutable_gpu_data();
			const Dtype* weight = this->blobs_[0]->gpu_data();
			int weight_offset = M_ * K_;
			int col_offset = K_ * mem_group_size;
			int result_offset = M_ * mem_group_size;
			int top_offset = M_ * N_;

			//For the first time running, intialize the entry point buffer for batched gemm
			if (!entry_initialized_){
				const Dtype** weight_entry_cpu_data = (const Dtype**)weight_entry_->mutable_cpu_data();
				Dtype** col_entry_cpu_data = (Dtype**)col_entry_->mutable_cpu_data();
				Dtype** result_entry_cpu_data = (Dtype**)result_entry_->mutable_cpu_data();

				for (int i = 0; i < N_; i++){
					weight_entry_cpu_data[i] = weight + weight_offset * i;
					col_entry_cpu_data[i] = col_data + col_offset * i;
					result_entry_cpu_data[i] = trans_data + result_offset * i;
				}
				entry_initialized_ = true;
			}

			// get gpu version of these entry points
			Dtype** weight_entry_gpu_data = (Dtype**)weight_entry_->mutable_gpu_data();
			Dtype** col_entry_gpu_data = (Dtype**)col_entry_->mutable_gpu_data();
			Dtype** result_entry_gpu_data = (Dtype**)result_entry_->mutable_gpu_data();

			//add bias to top first
			if (bias_term_)
			{
				// distribute bias into outputs
				caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
					num_, N_*(num_output_), 1,
					(Dtype)1., 
					reinterpret_cast<const Dtype*>(bias_multiplier_->gpu_data()),
					this->blobs_[1]->gpu_data(), 					
					(Dtype)0., top_data);
			}

			int mem_group_counter = 0;
			for (int n = 0; n < num_; n+=mem_group_size, mem_group_counter++){

				// im2col, here we attach sames columns from images inside a mem group to a contiguous block.
				size_t this_mem_group_size = min(mem_group_size,num_-n);

				bu_im2col_gpu(bottom_data + bottom[0]->offset(n), channels_, height_,
						width_, kernel_size_, pad_, stride_, col_data, this_mem_group_size, true);


				caffe_gpu_gemm_batched<Dtype>(CblasNoTrans, CblasNoTrans, 
										this_mem_group_size, M_, K_,
										(Dtype)1., (const Dtype **)col_entry_gpu_data, (const Dtype **)weight_entry_gpu_data, 
										(Dtype)0., result_entry_gpu_data, N_);

				// permute 3d and add bias
				cu_permute_3D_acc_gpu( trans_data, top_data + (*top)[0]->offset(n),
										M_, this_mem_group_size, N_,
										XYZtoZXY, Dtype(1.));
								
			}

			return Dtype(0.);
	}




	template <typename Dtype>
	void LocallyConnectedLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const bool propagate_down, vector<Blob<Dtype>*>* bottom) 	
	{
		//BUG FOUND! update memory before copy to slave memory!!!
		CUDA_CHECK(cudaDeviceSynchronize());
		//Caffe::switch_to_master_device();

		const Dtype* top_diff = top[0]->gpu_diff();
		Dtype* top_data = top[0]->mutable_gpu_data();
		const Dtype* weight = this->blobs_[0]->gpu_data();
		Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
		const Dtype* bottom_data = (*bottom)[0]->gpu_data();
		Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
		Dtype* col_data = col_buffer_.mutable_gpu_data();
		Dtype* col_diff = col_buffer_.mutable_gpu_diff();
		Dtype* trans_data = trans_buffer_.mutable_gpu_data();
		Dtype* trans_diff = trans_buffer_.mutable_gpu_diff();
		// bias gradient if necessary
		Dtype* bias_diff = NULL;


		int weight_offset = M_ * K_;
		int col_offset = K_ * mem_group_size;
		int result_offset = M_ * mem_group_size;
		int top_offset = M_ * N_;
		int bias_offset = num_output_;



		if (!bp_entry_initialized_)
		{
			Dtype **col_diff_entry_cpu_data = (Dtype **)col_diff_entry_->mutable_cpu_data();
			Dtype **weight_diff_entry_cpu_data = (Dtype **)weight_diff_entry_->mutable_cpu_data();

			for (int i = 0; i < N_; i++)
			{
				col_diff_entry_cpu_data[i] = col_diff + col_offset * i;
				weight_diff_entry_cpu_data[i] = weight_diff + weight_offset * i;
			}
			bp_entry_initialized_ = true;
		} 
	
		Dtype** weight_entry_gpu_data = (Dtype**)weight_entry_->mutable_gpu_data();
		Dtype** weight_diff_entry_gpu_data = (Dtype**)weight_diff_entry_->mutable_gpu_data();
		Dtype** col_entry_gpu_data = (Dtype**)col_entry_->mutable_gpu_data();
		Dtype** col_diff_entry_gpu_data = (Dtype**)col_diff_entry_->mutable_gpu_data();
		Dtype** trans_entry_gpu_data = (Dtype**)result_entry_->mutable_gpu_data();

		if (bias_term_)
		{
			bias_diff = this->blobs_[1]->mutable_gpu_diff();

			caffe_gpu_gemv<Dtype>(CblasTrans, num_, num_output_ * N_, 
						1.,	top_diff,
						reinterpret_cast<const Dtype*>(row_sumer_->gpu_data()),
						0., bias_diff);
		}


		CUDA_CHECK(cudaMemsetAsync(weight_diff, 0,
				sizeof(Dtype) * this->blobs_[0]->count(),
				Caffe::cu_stream()));

		int ct = this->blobs_[0]->count();

		int mem_group_counter = 0;
		for (int n = 0; n < num_; n+= mem_group_size, mem_group_counter++)
		{
			size_t this_mem_group_size = min(mem_group_size,num_-n);
			bu_im2col_gpu(bottom_data + (*bottom)[0]->offset(n), channels_, height_,
						width_, kernel_size_, pad_, stride_, col_data, this_mem_group_size, true);
				
			cu_permute_3D_acc_gpu(top_diff + top[0]->offset(n), trans_data, N_, M_, this_mem_group_size, XYZtoZYX);

			caffe_gpu_gemm_batched<Dtype>(CblasTrans, CblasTrans, 
				K_, M_, this_mem_group_size,
				(Dtype)1., (const Dtype **)col_entry_gpu_data, (const Dtype **)trans_entry_gpu_data,
				(Dtype)1., weight_diff_entry_gpu_data, N_);
			
			if (propagate_down)
			{
				caffe_gpu_gemm_batched<Dtype>(CblasNoTrans, CblasNoTrans, 
					K_, this_mem_group_size, M_,
					(Dtype)1., (const Dtype **)weight_entry_gpu_data, (const Dtype **)trans_entry_gpu_data,  
					(Dtype)0., col_diff_entry_gpu_data,  N_);

				bu_col2im_gpu(col_diff, channels_, height_, width_, kernel_size_, pad_,
					stride_, bottom_diff + (*bottom)[0]->offset(n), this_mem_group_size, true);
			}
		}		
		// Sync master device
		CUDA_CHECK(cudaStreamSynchronize(Caffe::cu_stream()));
	}


	INSTANTIATE_CLASS(LocallyConnectedLayer);

}  // namespace caffe
