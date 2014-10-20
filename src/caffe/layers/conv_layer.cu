// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"

#include <iostream>


namespace caffe {

	template <typename Dtype>
	Dtype ConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
			//BUG FOUND! update memory before copy to slave memory!!!
			CUDA_CHECK(cudaDeviceSynchronize());
			Caffe::switch_to_master_device();

			const Dtype* bottom_data = bottom[0]->gpu_data();
			Dtype* top_data = (*top)[0]->mutable_gpu_data();
			Dtype* col_data = col_buffer_.mutable_gpu_data();
			Dtype* bias_data = bias_buffer_.mutable_gpu_data();
			Dtype* trans_data = trans_buffer_.mutable_gpu_data();
			const Dtype* weight = this->blobs_[0]->gpu_data();
			int weight_offset = M_ * K_;
			int col_offset = K_ * N_;
			int top_offset = M_ * N_;

			Dtype* slave_bottom_data;
			Dtype* slave_top_data;
			Dtype* slave_col_data;
			Dtype* slave_trans_data;
			Dtype* slave_bias_data;
			Dtype* slave_weight_data;

			bool master_slave_flag = Caffe::gpu_mode() == Caffe::MASTER_SLAVE;

			int tot_group_num = (num_ + mem_group_size - 1) / mem_group_size;
			int mid_group_count = (tot_group_num+1) / 2;


			if (master_slave_flag)
			{
				Caffe::switch_to_slave_device();
				slave_bottom_data = (Dtype*)slave_bottom_buffer->mutable_gpu_data();
				slave_top_data = (Dtype*)slave_top_buffer->mutable_gpu_data();
				slave_col_data = (Dtype*)slave_col_buffer_->mutable_gpu_data();
				slave_bias_data = (Dtype*)slave_bias_buffer_->mutable_gpu_data();
				slave_trans_data = (Dtype*)slave_trans_buffer_->mutable_gpu_data();
				slave_weight_data = (Dtype*) slave_weight_buffer_->mutable_gpu_data();

				if (layer_stream.size()==0){
				// Here we initialize a series of streams for further accelerating the computation
					for (int i = 0; i<tot_group_num-mid_group_count;i++){
						layer_stream.push_back(cudaStream_t());
						CUDA_CHECK(cudaStreamCreate(&layer_stream[i]));
					}
				}
				Caffe::switch_to_master_device();
			}

			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
				N_, 1, (Dtype)1., this->blobs_[1]->gpu_data(),
				reinterpret_cast<const Dtype*>(bias_multiplier_->gpu_data()),
				(Dtype)0., bias_data);


			
			
			int mem_group_counter = 0;

			if (master_slave_flag)
			{
				//copy weight and bias to slave device
				CUDA_CHECK(cudaMemcpyPeerAsync( 
					slave_weight_data, Caffe::slave_device_id(),
					weight, Caffe::master_device_id(),
					num_output_* K_ * sizeof(Dtype),
					Caffe::slave_cu_stream()
					));
				CUDA_CHECK(cudaMemcpyPeerAsync( 
					slave_bias_data, Caffe::slave_device_id(),
					bias_data, Caffe::master_device_id(),
					num_output_* N_ * sizeof(Dtype),
					Caffe::slave_cu_stream()
					));

				//copy bottom data to slave device by mem_group
				for (int i = 0; i< tot_group_num - mid_group_count;i++){
						CUDA_CHECK(cudaMemcpyPeerAsync(
					slave_bottom_data + bottom[0]->offset((mid_group_count + i) * mem_group_size ) , Caffe::slave_device_id(),
					bottom_data + bottom[0]->offset((mid_group_count + i) * mem_group_size), Caffe::master_device_id(),
					bottom[0]->offset(min(num_ - (i + mid_group_count) * mem_group_size, mem_group_size)) * sizeof(Dtype)
					, layer_stream[i]
					));
				}
				
			}


			for (int n = 0; n < num_; n+=mem_group_size, mem_group_counter++) {

				size_t this_mem_group_size = min(mem_group_size,num_-n);

				if (mem_group_counter == mid_group_count && master_slave_flag){
					Caffe::switch_to_slave_device();

					// Make sure weights are copied to slave device
					CUDA_CHECK(cudaStreamSynchronize(Caffe::slave_cu_stream()));
				}

				if (mem_group_counter >= mid_group_count && master_slave_flag)
				{
					

					Caffe::set_cu_stream(layer_stream[mem_group_counter - mid_group_count]);


					// First, im2col
					bu_im2col_gpu(slave_bottom_data + bottom[0]->offset(n), channels_, height_,
						width_, kernel_size_, pad_, stride_, slave_col_data, this_mem_group_size);

					// Second, innerproduct with groups
					for (int g = 0; g < group_; ++g) {
						caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,  
							M_, N_*this_mem_group_size, K_,
							(Dtype)1., slave_weight_data + weight_offset * g,
							slave_col_data + col_offset * g * this_mem_group_size,  
							(Dtype)0., slave_trans_data + top_offset * g * this_mem_group_size);
					}

					cu_mat2im_c_gpu(slave_trans_data, num_output_, N_, slave_top_data + (*top)[0]->offset(n), (bias_term_)?(Dtype)1.:(Dtype)0., slave_bias_data, this_mem_group_size);
					
					// Sync before next operation to secure that buffer is released
					CUDA_CHECK(cudaStreamSynchronize(layer_stream[mem_group_counter - mid_group_count]));

					// Copy back result to master device, we overlap data transfer and computation
					CUDA_CHECK(cudaMemcpyPeerAsync( 
						top_data + (*top)[0]->offset(n), Caffe::master_device_id(),
						slave_top_data + (*top)[0]->offset(n), Caffe::slave_device_id(),
						(*top)[0]->offset(this_mem_group_size)*sizeof(Dtype)
						, Caffe::slave_cu_stream()));

				}
				else
				{
					bu_im2col_gpu(bottom_data + bottom[0]->offset(n), channels_, height_,
						width_, kernel_size_, pad_, stride_, col_data, this_mem_group_size);

					// Second, innerproduct with groups
					for (int g = 0; g < group_; ++g) {
						caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,  
							M_, N_*this_mem_group_size, K_,
							(Dtype)1., weight + weight_offset * g,
							col_data + col_offset * g * this_mem_group_size,  
							(Dtype)0., trans_data + top_offset * g * this_mem_group_size);
					}
					// third, add bias
					cu_mat2im_c_gpu(trans_data, num_output_, N_, top_data + (*top)[0]->offset(n), (bias_term_)?(Dtype)1.:(Dtype)0., bias_data, this_mem_group_size);
				}
			}

			if (master_slave_flag)
			{
				

				Caffe::switch_to_master_device();
			}

			// Sync master device
			CUDA_CHECK(cudaStreamSynchronize(Caffe::cu_stream()));
			if (master_slave_flag){
				// Sync slave device
				CUDA_CHECK(cudaStreamSynchronize(Caffe::slave_cu_stream()));
			}

			return Dtype(0.);
	}




	template <typename Dtype>
	void ConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
			//BUG FOUND! update memory before copy to slave memory!!!
			CUDA_CHECK(cudaDeviceSynchronize());
			Caffe::switch_to_master_device();

			const Dtype* top_diff = top[0]->gpu_diff();
			Dtype* top_data = top[0]->mutable_gpu_data();
			const Dtype* weight = this->blobs_[0]->gpu_data();
			Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
			const Dtype* bottom_data = (*bottom)[0]->gpu_data();
			Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
			Dtype* col_data = col_buffer_.mutable_gpu_data();
			Dtype* col_diff = col_buffer_.mutable_gpu_diff();
			Dtype* trans_data = trans_buffer_.mutable_gpu_data();
			// bias gradient if necessary
			Dtype* bias_diff = NULL;

			// Set of slave buffer pointers
			Dtype* slave_bottom_data;
			Dtype* slave_top_diff;
			Dtype* slave_col_data;
			Dtype* slave_trans_data;
			Dtype* slave_weight_data;
			Dtype* slave_col_diff;
			Dtype* slave_weight_diff;
			Dtype* slave_to_master_weight_diff;

			// Computation for bias diff is small, do it on master device
			if (bias_term_) {
				bias_diff = this->blobs_[1]->mutable_gpu_diff();

				caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output_ * num_, N_,
						1., top_diff,
						reinterpret_cast<const Dtype*>(bias_multiplier_->gpu_data()),
						0., top_data);
				caffe_gpu_gemv<Dtype>(CblasTrans, num_, num_output_, 
						1.,	top_data,
						reinterpret_cast<const Dtype*>(row_sumer_->gpu_data()),
						0., bias_diff);

			}

			int weight_offset = M_ * K_;
			int col_offset = K_ * N_;
			int top_offset = M_ * N_;
			CUDA_CHECK(cudaMemsetAsync(weight_diff, 0,
				sizeof(Dtype) * this->blobs_[0]->count(),
				Caffe::cu_stream()));

			int tot_group_num = (num_ + mem_group_size - 1) / mem_group_size;
			int mid_group_count = (tot_group_num+1) / 2;

			int mem_group_counter = 0;

			bool master_slave_flag = Caffe::gpu_mode() == Caffe::MASTER_SLAVE;

			if (master_slave_flag)
			{
				slave_to_master_weight_diff = (Dtype*)master_weight_diff_buffer_->mutable_gpu_data();
				CUDA_CHECK(cudaMemsetAsync(slave_to_master_weight_diff, 0,
					sizeof(Dtype) * this->blobs_[0]->count(),
					Caffe::cu_stream()));

				Caffe::switch_to_slave_device();
				slave_bottom_data = (Dtype*)slave_bottom_buffer->mutable_gpu_data();
				slave_top_diff = (Dtype*)slave_top_buffer->mutable_gpu_data();
				slave_col_data = (Dtype*)slave_col_buffer_->mutable_gpu_data();
				slave_trans_data = (Dtype*)slave_trans_buffer_->mutable_gpu_data();
				slave_weight_data = (Dtype*) slave_weight_buffer_->mutable_gpu_data();
				slave_weight_diff = (Dtype*) slave_weight_diff_buffer_->mutable_gpu_data();
				slave_col_diff = (Dtype*) slave_col_diff_buffer_->mutable_gpu_data();

				CUDA_CHECK(cudaMemsetAsync(slave_weight_diff, 0,
					sizeof(Dtype) * this->blobs_[0]->count(),
					Caffe::slave_cu_stream()));
				
				Caffe::switch_to_master_device();

				//No need to copy bottom and weight data to slave device, it's copied in forward pass

				//Copy top diff data to slave device, note here we also devide mem_group
				for (int i = 0; i< tot_group_num - mid_group_count;i++){
						CUDA_CHECK(cudaMemcpyPeerAsync(
					slave_top_diff + top[0]->offset((mid_group_count + i) * mem_group_size ) , Caffe::slave_device_id(),
					top_diff + top[0]->offset((mid_group_count + i) * mem_group_size), Caffe::master_device_id(),
					top[0]->offset(min(num_ - (i + mid_group_count) * mem_group_size, mem_group_size)) * sizeof(Dtype)
					, layer_stream[i]
					));
				}
			}

			for (int n = 0; n < num_; n+=mem_group_size, mem_group_counter++) {
				// since we saved memory in the forward pass by not storing all col data,
				// we will need to recompute them.
				size_t this_mem_group_size = min(mem_group_size,num_-n);

				if (mem_group_counter == mid_group_count && master_slave_flag){
					Caffe::switch_to_slave_device();
					CUDA_CHECK(cudaStreamSynchronize(Caffe::slave_cu_stream())); //confirm the weights are transfered to slave device
				}

				
				if (mem_group_counter >= mid_group_count && master_slave_flag)
				{
					Caffe::set_cu_stream(layer_stream[mem_group_counter - mid_group_count]);

					// Do im2col
					bu_im2col_gpu(slave_bottom_data + (*bottom)[0]->offset(n), channels_, height_,
						width_, kernel_size_, pad_, stride_, slave_col_data, this_mem_group_size);

					// Reshape memory so that every row of each top image are contiguous
					cu_im2mat_gpu(slave_top_diff + top[0]->offset(n),1, num_output_, N_, 
						slave_trans_data,
						this_mem_group_size);

					// Do inner product to accumulate weight difference
					for (int g = 0; g < group_; ++g) {
						caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_*this_mem_group_size,
							(Dtype)1., slave_trans_data + top_offset * g * this_mem_group_size,
							slave_col_data + col_offset * g * this_mem_group_size,  (Dtype)1.,
							slave_weight_diff + weight_offset * g);
					}
					// gradient w.r.t. bottom data, if necessary
					if (propagate_down) {
						for (int g = 0; g < group_; ++g) {
							caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_ * this_mem_group_size, M_,
								(Dtype)1., slave_weight_data + weight_offset * g,
								slave_trans_data + top_offset * g * this_mem_group_size,
								(Dtype)0., slave_col_diff + col_offset * g * this_mem_group_size);
						}
						// col2im back to the data
						// slave_bottom_data holds slave_bottom_diff, in order to save memory
						bu_col2im_gpu_rot(slave_col_diff, channels_, height_, width_, kernel_size_, pad_,
							stride_, slave_bottom_data + (*bottom)[0]->offset(n), this_mem_group_size);

						// Sync before next mem_group because they will use the same buffers
						CUDA_CHECK(cudaStreamSynchronize(layer_stream[mem_group_counter - mid_group_count]));

						// Once synced, send the result to master gpu, thus we overlap data transfer and calculation
						CUDA_CHECK(cudaMemcpyPeerAsync( 
							bottom_diff + (*bottom)[0]->offset(n), Caffe::master_device_id(),
							slave_bottom_data + (*bottom)[0]->offset(n), Caffe::slave_device_id(),
							(*bottom)[0]->offset(this_mem_group_size)*sizeof(Dtype)
							, Caffe::slave_cu_stream()));
					}else{

						// Sync before next mem_group because they will use the same buffers
						CUDA_CHECK(cudaStreamSynchronize(layer_stream[mem_group_counter - mid_group_count]));
					}



				}
				else
				{
					bu_im2col_gpu(bottom_data + (*bottom)[0]->offset(n), channels_, height_,
						width_, kernel_size_, pad_, stride_, col_data, this_mem_group_size);

					cu_im2mat_gpu(top_diff + top[0]->offset(n),1, num_output_, N_, 
						trans_data,
						this_mem_group_size);
					// gradient w.r.t. weight. Note that we will accumulate diffs.
					for (int g = 0; g < group_; ++g) {
						caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_*this_mem_group_size,
							(Dtype)1., trans_data + top_offset * g * this_mem_group_size,
							col_data + col_offset * g * this_mem_group_size,  (Dtype)1.,
							weight_diff + weight_offset * g);
					}
					// gradient w.r.t. bottom data, if necessary
					if (propagate_down) {
						for (int g = 0; g < group_; ++g) {
							caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_ * this_mem_group_size, M_,
								(Dtype)1., weight + weight_offset * g,
								trans_data + top_offset * g * this_mem_group_size,
								(Dtype)0., col_diff + col_offset * g * this_mem_group_size);
						}
						// col2im back to the data
						bu_col2im_gpu_rot(col_diff, channels_, height_, width_, kernel_size_, pad_,
							stride_, bottom_diff + (*bottom)[0]->offset(n), this_mem_group_size);
					}
				}
			}

			if (master_slave_flag)
			{
				// Sync last mem_group to prepare sending weight_diff to master device
				CUDA_CHECK(cudaStreamSynchronize(layer_stream.back()));

				// Send accumulated weight_diff to master gpu for reducing
				CUDA_CHECK(cudaMemcpyPeerAsync( 
					slave_to_master_weight_diff, Caffe::master_device_id(),
					slave_weight_diff, Caffe::slave_device_id(),
					sizeof(Dtype) * this->blobs_[0]->count()
					, Caffe::slave_cu_stream()));

				Caffe::switch_to_master_device();

			}

			// Sync master device
			CUDA_CHECK(cudaStreamSynchronize(Caffe::cu_stream()));
			if (master_slave_flag)
			{

				CUDA_CHECK(cudaStreamSynchronize(Caffe::slave_cu_stream()));

				// Sum two part of weigth diff and get weight updates
				caffe_gpu_axpy<Dtype>(this->blobs_[0]->count(), (Dtype)1.0, slave_to_master_weight_diff, weight_diff);

				// Sync master device
				CUDA_CHECK(cudaStreamSynchronize(Caffe::cu_stream()));
			}
	}


	INSTANTIATE_CLASS(ConvolutionLayer);

}  // namespace caffe
