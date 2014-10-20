// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include "caffe/common.hpp"
#include "caffe/util/im2col.hpp"

namespace caffe {

	template <typename Dtype>
__global__ void bu_im2col_gpu_kernel(
	const int n, const Dtype* data_im,
	const int height, const int width, const int ksize, const int pad,
	const int stride, const int height_col, const int width_col,
	Dtype* data_col,
	const int data_im_size,
	const int data_col_size,
	const int batch_size,
	const bool trans) 
{
	/*for(int batch_index = 0; batch_index < batch_size; batch_index++)
	{
		for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < n; index += blockDim.x * gridDim.x){
			int w_out = index % width_col;
			int h_index = index / width_col;
			int h_out = h_index % height_col;
			int channel_in = h_index / height_col;
			int channel_out = channel_in * ksize * ksize;
			int h_in = h_out * stride - pad;
			int w_in = w_out * stride - pad;
			Dtype* data_col_ptr = data_col;
			data_col_ptr += batch_index* data_col_size + (channel_out * height_col + h_out) * width_col + w_out;
			const Dtype* data_im_ptr = data_im;
			data_im_ptr += batch_index* data_im_size + (channel_in * height + h_in) * width + w_in;

			for (int i = 0; i < ksize; ++i) {
				for (int j = 0; j < ksize; ++j) {
					int h = h_in + i;
					int w = w_in + j;
					*data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ?
						data_im_ptr[i * width + j]  : 0;
					data_col_ptr += height_col * width_col;
				}
			}

		}
	}*/
	int N = height_col * width_col;
	int channels_all = n / width_col / height_col;
	int trans_col_offset = ksize * ksize * channels_all;

	for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < n; index += blockDim.x * gridDim.x){
		int w_out = index % width_col;
		int h_index = index / width_col;
		int h_out = h_index % height_col;
		int channel_in = h_index / height_col;
		int channel_out = channel_in * ksize * ksize;
		int h_in = h_out * stride - pad;
		int w_in = w_out * stride - pad;

		if (!trans){
			Dtype* data_col_ptr = data_col;
			data_col_ptr += channel_out * N * batch_size + h_out * width_col + w_out;

			const Dtype* data_im_ptr = data_im;
			data_im_ptr += (channel_in * height + h_in) * width + w_in;


			for(int batch_index = 0; batch_index < batch_size; batch_index++)
			{
				Dtype* data_write_col_ptr = data_col_ptr;

				for (int i = 0; i < ksize; ++i) {
					for (int j = 0; j < ksize; ++j) {
						int h = h_in + i;
						int w = w_in + j;
						*data_write_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ?
							data_im_ptr[i * width + j]  : 0;
						data_write_col_ptr += N * batch_size;
					}
				}

				data_col_ptr += N;
				data_im_ptr += data_im_size;
			}
		}else{	

			Dtype* data_col_ptr = data_col;
			data_col_ptr += (h_out * width_col + w_out) * batch_size * trans_col_offset + channel_in * ksize * ksize;
				//index * batch_size * ksize * ksize;

			const Dtype* data_im_ptr = data_im;
			data_im_ptr += (channel_in * height + h_in) * width + w_in;

			for(int batch_index = 0; batch_index < batch_size; batch_index++)
			{
				Dtype* data_write_col_ptr = data_col_ptr;

				for (int i = 0; i < ksize; ++i) {
					for (int j = 0; j < ksize; ++j) {
						int h = h_in + i;
						int w = w_in + j;
						*data_write_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ?
							data_im_ptr[i * width + j]  : 0;
						data_write_col_ptr ++;
					}
				}

				data_col_ptr += trans_col_offset;
				data_im_ptr += data_im_size;
			}
		}
	}
}


template <typename Dtype>
void bu_im2col_gpu(const Dtype* data_im, const int channels,
				   const int height, const int width, const int ksize, const int pad,
				   const int stride, Dtype* data_col, const int batch_size, const bool trans){
	// We are going to launch channels * height_col * width_col kernels, each
	// kernel responsible for copying a single-channel grid.
	int height_col = (height + 2 * pad - ksize) / stride + 1;
	int width_col = (width + 2 * pad - ksize) / stride + 1;
	int num_kernels = channels * height_col * width_col;

	int data_im_size = height*width*channels;
	int data_col_size = num_kernels*ksize*ksize;

	// NOLINT_NEXT_LINE(whitespace/operators)
	bu_im2col_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels), // num_kernels/16, means each thread process 16 elements
		CAFFE_CUDA_NUM_THREADS, 0, Caffe::get_current_cu_stream()>>>(
		num_kernels, data_im, height, width, ksize, pad, stride, height_col,
		width_col, data_col, data_im_size, data_col_size, batch_size, trans);
	CUDA_POST_KERNEL_CHECK;
}

// Explicit instantiation
template void bu_im2col_gpu<float>(
	const float* data_im, const int channels,
	const int height, const int width, const int ksize, const int pad,
	const int stride, float* data_col,
	const int batch_size, const bool trans);
template void bu_im2col_gpu<double>(
	const double* data_im, const int channels,
	const int height, const int width, const int ksize, const int pad,
	const int stride, double* data_col,
	const int batch_size, const bool trans);

template <typename Dtype>
void bu_im2col_gpu(const Dtype* data_im, const int channels,
				   const int height, const int width, const int ksize, const int pad,
				   const int stride, Dtype* data_col, const int batch_size)
{
	bu_im2col_gpu(data_im, channels,
					height, width, ksize, pad,
					stride, data_col, batch_size, false);
}





// Explicit instantiation
template void bu_im2col_gpu<float>(
	const float* data_im, const int channels,
	const int height, const int width, const int ksize, const int pad,
	const int stride, float* data_col,
	const int batch_size);
template void bu_im2col_gpu<double>(
	const double* data_im, const int channels,
	const int height, const int width, const int ksize, const int pad,
	const int stride, double* data_col,
	const int batch_size);


template <typename Dtype>
__global__ void bu_im2col_gpu_kernel_rot(
	const int n, const Dtype* data_im,
	const int height, const int width, const int ksize, const int pad,
	const int stride, const int height_col, const int width_col,
	Dtype* data_col,
	const int data_im_size,
	const int data_col_size,
	const int batch_size) 
{
	for(int batch_index = 0; batch_index < batch_size; batch_index++)
	{
		for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < n; index += blockDim.x * gridDim.x){
			int w_out = index % width_col;
			int h_index = index / width_col;
			int h_out = h_index % height_col;
			int channel_in = h_index / height_col;
			int channel_out = channel_in * ksize * ksize;
			int h_in = h_out * stride - pad;
			int w_in = w_out * stride - pad;
			Dtype* data_col_ptr = data_col;
			data_col_ptr += batch_index* data_col_size + (channel_out * height_col + h_out) * width_col + w_out;
			const Dtype* data_im_ptr = data_im;
			data_im_ptr += batch_index* data_im_size + (channel_in * height + h_in) * width + w_in;

			for (int i = 0; i < ksize; ++i) {
				for (int j = 0; j < ksize; ++j) {
					int h = h_in + i;
					int w = w_in + j;
					*data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ?
						data_im_ptr[i * width + j]  : 0;
					data_col_ptr += height_col * width_col;
				}
			}

		}
	}
}

template <typename Dtype>
void bu_im2col_gpu_rot(const Dtype* data_im, const int channels,
				   const int height, const int width, const int ksize, const int pad,
				   const int stride, Dtype* data_col, const int batch_size)
{
	// We are going to launch channels * height_col * width_col kernels, each
	// kernel responsible for copying a single-channel grid.
	int height_col = (height + 2 * pad - ksize) / stride + 1;
	int width_col = (width + 2 * pad - ksize) / stride + 1;
	int num_kernels = channels * height_col * width_col;

	int data_im_size = height*width*channels;
	int data_col_size = num_kernels*ksize*ksize;
	// NOLINT_NEXT_LINE(whitespace/operators)
	bu_im2col_gpu_kernel_rot<Dtype><<<CAFFE_GET_BLOCKS(num_kernels), // num_kernels/16, means each thread process 16 elements
		CAFFE_CUDA_NUM_THREADS, 0, Caffe::get_current_cu_stream()>>>(
		num_kernels, data_im, height, width, ksize, pad, stride, height_col,
		width_col, data_col, data_im_size, data_col_size, batch_size);
	CUDA_POST_KERNEL_CHECK;
}

// Explicit instantiation
template void bu_im2col_gpu_rot<float>(
	const float* data_im, const int channels,
	const int height, const int width, const int ksize, const int pad,
	const int stride, float* data_col,
	const int batch_size);
template void bu_im2col_gpu_rot<double>(
	const double* data_im, const int channels,
	const int height, const int width, const int ksize, const int pad,
	const int stride, double* data_col,
	const int batch_size);


template <typename Dtype>
__global__ void im2col_gpu_kernel(const int n, const Dtype* data_im,
    const int height, const int width, const int ksize, const int pad,
    const int stride, const int height_col, const int width_col,
    Dtype* data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    int w_out = index % width_col;
    index /= width_col;
    int h_out = index % height_col;
    int channel_in = index / height_col;
    int channel_out = channel_in * ksize * ksize;
    int h_in = h_out * stride - pad;
    int w_in = w_out * stride - pad;
    data_col += (channel_out * height_col + h_out) * width_col + w_out;
    data_im += (channel_in * height + h_in) * width + w_in;
    for (int i = 0; i < ksize; ++i) {
      for (int j = 0; j < ksize; ++j) {
        int h = h_in
			+ i;
        int w = w_in + j;
        *data_col = (h >= 0 && w >= 0 && h < height && w < width) ?
            data_im[i * width + j] : 0;
        data_col += height_col * width_col;
      }
    }
  }
}

template <typename Dtype>
void im2col_gpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int ksize, const int pad,
    const int stride, Dtype* data_col) {
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int height_col = (height + 2 * pad - ksize) / stride + 1;
  int width_col = (width + 2 * pad - ksize) / stride + 1;
  int num_kernels = channels * height_col * width_col;
  // NOLINT_NEXT_LINE(whitespace/operators)
  im2col_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, data_im, height, width, ksize, pad, stride, height_col,
      width_col, data_col);
  CUDA_POST_KERNEL_CHECK;
}


// Explicit instantiation
template void im2col_gpu<float>(const float* data_im, const int channels,
    const int height, const int width, const int ksize, const int pad,
    const int stride, float* data_col);
template void im2col_gpu<double>(const double* data_im, const int channels,
    const int height, const int width, const int ksize, const int pad,
    const int stride, double* data_col);

template <typename Dtype>
__global__ void col2im_gpu_kernel(const int n, const Dtype* data_col,
    const int height, const int width, const int channels, const int ksize,
    const int pad, const int stride, const int height_col, const int width_col,
    Dtype* data_im) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype val = 0;
    int w = index % width + pad;
    int h = (index / width) % height + pad;
    int c = index / (width * height);
    // compute the start and end of the output
    int w_col_start = (w < ksize) ? 0 : (w - ksize) / stride + 1;
    int w_col_end = min(w / stride + 1, width_col);
    int h_col_start = (h < ksize) ? 0 : (h - ksize) / stride + 1;
    int h_col_end = min(h / stride + 1, height_col);
    /*
    for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
      for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
        // the col location: [c * width * height + h_out, w_out]
        int c_col = c * ksize * ksize + (h - h_col * stride) * ksize + (w - w_col * stride);
        val += data_col[(c_col * height_col + h_col) * width_col + w_col];
      }
    }
    */
    // equivalent implementation
    int offset = (c * ksize * ksize + h * ksize + w) * height_col * width_col;
    int coeff_h_col = (1 - stride * ksize * height_col) * width_col;
    int coeff_w_col = (1 - stride * height_col * width_col);
    for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
      for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
        val += data_col[offset + h_col * coeff_h_col + w_col * coeff_w_col];
      }
    }
    data_im[index] = val;
  }
}

template <typename Dtype>
void col2im_gpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int ksize, const int pad,
    const int stride, Dtype* data_im) {
  // CUDA_CHECK(cudaMemset(data_im, 0,
  //            sizeof(Dtype) * height * width * channels));
  int height_col = (height + 2 * pad - ksize) / stride + 1;
  int width_col = (width + 2 * pad - ksize) / stride + 1;
  int num_kernels = channels * height * width;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  // NOLINT_NEXT_LINE(whitespace/operators)
  col2im_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, data_col, height, width, channels, ksize, pad, stride,
      height_col, width_col, data_im);
  CUDA_POST_KERNEL_CHECK;
}


// Explicit instantiation
template void col2im_gpu<float>(const float* data_col, const int channels,
    const int height, const int width, const int psize, const int pad,
    const int stride, float* data_im);
template void col2im_gpu<double>(const double* data_col, const int channels,
    const int height, const int width, const int psize, const int pad,
    const int stride, double* data_im);



//Enable batched col2im
template <typename Dtype>
__global__ void bu_col2im_gpu_kernel(const int n, const Dtype* data_col,
    const int height, const int width, const int channels, const int ksize,
    const int pad, const int stride, const int height_col, const int width_col,
    Dtype* data_im,
	const int batch_size, bool trans) 
{
	CUDA_KERNEL_LOOP(index, n) 
	{

		//
		int col_length = channels*ksize*ksize;
		int col_offset = height_col*width_col; // offset per col image
		int im_offset = n;
		int t_index = index;
		int col_start = 0;

		int w = index % width + pad;
		int h = (index / width) % height + pad;
		int c = index / (width * height);
		// compute the start and end of the output
		int w_col_start = (w < ksize) ? 0 : (w - ksize) / stride + 1;
		int w_col_end = min(w / stride + 1, width_col);
		int h_col_start = (h < ksize) ? 0 : (h - ksize) / stride + 1;
		int h_col_end = min(h / stride + 1, height_col);

		if (!trans)
		{
			for (int batch_index = 0; batch_index<batch_size; batch_index++)
			{
				Dtype val = 0;
				/*
				for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
				for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
				// the col location: [c * width * height + h_out, w_out]
				int c_col = c * ksize * ksize + (h - h_col * stride) * ksize + (w - w_col * stride);
				val += data_col[(c_col * height_col + h_col) * width_col + w_col];
				}
				}
				*/
				// equivalent implementation


				int offset = (c * ksize * ksize + h * ksize + w) * height_col * width_col+col_start;
				int coeff_h_col = (1 - stride * ksize * height_col) * width_col;
				int coeff_w_col = (1 - stride * height_col * width_col);
				for (int h_col = h_col_start; h_col < h_col_end; ++h_col) 
				{
					for (int w_col = w_col_start; w_col < w_col_end; ++w_col) 
					{
						val += data_col[offset + h_col * coeff_h_col + w_col * coeff_w_col];
					}
				}
				data_im[t_index] = val;
				t_index += n;
				col_start += col_offset;
			}
		}
		else
		{
			for (int batch_index = 0; batch_index < batch_size; batch_index++)
			{
				Dtype val = 0;
			//	for (int h_col = h_col_start; h_col < h_col_end; ++h_col)
			//	{
			//		for (int w_col = w_col_start; w_col < w_col_end; ++w_col)
			//		{
			//			int N_index = weight_col * h_col + w_col;
			//			int k_index = c * ksize * ksize + (h - h_col * stride)* ksize + w - stride * w_col;

			//			int col_index = batch_index * col_offset + N_index * batch_size * col_offset + k_index;
			//			val += data_col[col_index];
			//		}
			//	}

				int offset = col_start + (c * ksize *ksize + h * ksize + w)* batch_size;
				int coeff_h_col = (width_col  - ksize  * stride) * batch_size * col_length;
				int coeff_w_col = (col_length -  stride) * batch_size;
				for (int h_col = h_col_start; h_col < h_col_end; ++h_col)
				{
					for (int w_col = w_col_start; w_col < w_col_end; ++w_col)
					{
						val += data_col[offset + coeff_h_col * h_col + coeff_w_col * w_col];
					}
				}

				data_im[t_index] = val;
				t_index += n;
				col_start += 1;
			}

		}
	}
}

template <typename Dtype>
void bu_col2im_gpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int ksize, const int pad,
    const int stride, Dtype* data_im,
	const int batch_size) {
  // CUDA_CHECK(cudaMemset(data_im, 0,
  //            sizeof(Dtype) * height * width * channels));
  int height_col = (height + 2 * pad - ksize) / stride + 1;
  int width_col = (width + 2 * pad - ksize) / stride + 1;
  int num_kernels = channels * height * width;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  // NOLINT_NEXT_LINE(whitespace/operators)
  bu_col2im_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS, 0, Caffe::get_current_cu_stream() >>>(
      num_kernels, data_col, height, width, channels, ksize, pad, stride,
      height_col, width_col, data_im,
	  batch_size, false);
  CUDA_POST_KERNEL_CHECK;
}


// Explicit instantiation
template void bu_col2im_gpu<float>(const float* data_col, const int channels,
    const int height, const int width, const int psize, const int pad,
    const int stride, float* data_im,
	const int batch_size);
template void bu_col2im_gpu<double>(const double* data_col, const int channels,
    const int height, const int width, const int psize, const int pad,
    const int stride, double* data_im,
	const int batch_size);

template <typename Dtype>
void bu_col2im_gpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int ksize, const int pad,
    const int stride, Dtype* data_im,
	const int batch_size, bool trans) {
  // CUDA_CHECK(cudaMemset(data_im, 0,
  //            sizeof(Dtype) * height * width * channels));
  int height_col = (height + 2 * pad - ksize) / stride + 1;
  int width_col = (width + 2 * pad - ksize) / stride + 1;
  int num_kernels = channels * height * width;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  // NOLINT_NEXT_LINE(whitespace/operators)
  bu_col2im_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS, 0, Caffe::get_current_cu_stream() >>>(
      num_kernels, data_col, height, width, channels, ksize, pad, stride,
      height_col, width_col, data_im,
	  batch_size, trans);
  CUDA_POST_KERNEL_CHECK;
}

template void bu_col2im_gpu<float>(const float* data_col, const int channels,
    const int height, const int width, const int psize, const int pad,
    const int stride, float* data_im,
	const int batch_size, bool trans);
template void bu_col2im_gpu<double>(const double* data_col, const int channels,
    const int height, const int width, const int psize, const int pad,
    const int stride, double* data_im,
	const int batch_size, bool trans);


//Enable batched col2im
template <typename Dtype>
__global__ void bu_col2im_gpu_rot_kernel(const int n, const Dtype* data_col,
    const int height, const int width, const int channels, const int ksize,
    const int pad, const int stride, const int height_col, const int width_col,
    Dtype* data_im,
	const int batch_size) {
  CUDA_KERNEL_LOOP(index, n) {
    
	//
	int col_length = channels*ksize*ksize;
	int col_offset = height_col*width_col; // offset per col image
	int im_offset = n;
	int t_index = index;

    int w = index % width + pad;
    int h = (index / width) % height + pad;
    int c = index / (width * height);
    // compute the start and end of the output
    int w_col_start = (w < ksize) ? 0 : (w - ksize) / stride + 1;
    int w_col_end = min(w / stride + 1, width_col);
    int h_col_start = (h < ksize) ? 0 : (h - ksize) / stride + 1;
    int h_col_end = min(h / stride + 1, height_col);

    // every batch, offset height_col * width_col
	int offset = (c * ksize * ksize + h * ksize + w) * height_col * width_col*batch_size;
	int coeff_h_col = (1 - stride * ksize * height_col*batch_size) * width_col;
	int coeff_w_col = (1 - stride * height_col * width_col*batch_size);
	for (int batch_index = 0; batch_index<batch_size; batch_index++){
		Dtype val = 0;
		
		for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
			for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
				val += data_col[offset + h_col * coeff_h_col + w_col * coeff_w_col];
			}
		}
		data_im[t_index] = val;
		t_index += n;
		offset += col_offset;
	}
  }
}

template <typename Dtype>
void bu_col2im_gpu_rot(const Dtype* data_col, const int channels,
    const int height, const int width, const int ksize, const int pad,
    const int stride, Dtype* data_im,
	const int batch_size) {
  // CUDA_CHECK(cudaMemset(data_im, 0,
  //            sizeof(Dtype) * height * width * channels));
  int height_col = (height + 2 * pad - ksize) / stride + 1;
  int width_col = (width + 2 * pad - ksize) / stride + 1;
  int num_kernels = channels * height * width;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  // NOLINT_NEXT_LINE(whitespace/operators)
  bu_col2im_gpu_rot_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS, 0, Caffe::get_current_cu_stream()>>>(
      num_kernels, data_col, height, width, channels, ksize, pad, stride,
      height_col, width_col, data_im,
	  batch_size);
  CUDA_POST_KERNEL_CHECK;
}


// Explicit instantiation
template void bu_col2im_gpu_rot<float>(const float* data_col, const int channels,
    const int height, const int width, const int psize, const int pad,
    const int stride, float* data_im,
	const int batch_size);
template void bu_col2im_gpu_rot<double>(const double* data_col, const int channels,
    const int height, const int width, const int psize, const int pad,
    const int stride, double* data_im,
	const int batch_size);

//Composite contigious images to one row-major matrix (long row)
template <typename Dtype>
__global__ void cu_im2mat_gpu_kernel(const int n, const Dtype* data_im,
    const int height, const int width, const int channels,
    Dtype* data_mat,
	const int batch_size) {
  CUDA_KERNEL_LOOP(index, n) {
    int temp = index;
	int col_now = temp % width;
	temp /= width;
	int row_now = temp % height;
	int ch_now = temp / height;

	int im_offset = n;
	int mat_offset = width;

	const Dtype *im_ptr = data_im + index;
	Dtype *mat_ptr = data_mat + col_now + (row_now + ch_now * height) * width * batch_size;


	for (int batch_index = 0; batch_index < batch_size; batch_index++){
		*(mat_ptr) = *(im_ptr);

		//offset both indexers
		im_ptr += im_offset;
		mat_ptr += mat_offset;
	}

	
  }
}

template <typename Dtype>
void cu_im2mat_gpu(const Dtype* data_im, const int channels,
    const int height, const int width, 
	Dtype* data_mat,
	const int batch_size) {
  // CUDA_CHECK(cudaMemset(data_im, 0,
  //            sizeof(Dtype) * height * width * channels));
  int num_kernels = channels * height * width;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  // NOLINT_NEXT_LINE(whitespace/operators)
  cu_im2mat_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS, 0, Caffe::get_current_cu_stream()>>>(
      num_kernels, data_im, height, width, channels, 
	  data_mat,
	  batch_size);
  CUDA_POST_KERNEL_CHECK;
}


// Explicit instantiation
template void cu_im2mat_gpu<float>(const float* data_im, const int channels,
    const int height, const int width, 
	float* data_mat,
	const int batch_size);
template void cu_im2mat_gpu<double>(const double* data_im, const int channels,
    const int height, const int width, 
	double* data_mat,
	const int batch_size);


//Convert composited big matrix (long row) to row-major images with carry
//Carry is added by A + beta * C
template <typename Dtype>
__global__ void cu_mat2im_c_gpu_kernel(const int n, const Dtype* data_mat,
    const int mat_height, const int mat_width,
    Dtype* data_im,
	Dtype beta,
	Dtype* data_carry,
	const int batch_size) {
  CUDA_KERNEL_LOOP(index, n) {
	int col_now = index % mat_width;
	int row_now = index / mat_width;

	int im_idx = index;
	int mat_idx = col_now + row_now * mat_width * batch_size;

	Dtype carry = data_carry[index] * beta;

	int im_offset = n;
	int mat_offset = mat_width;
	
	for (int batch_index = 0; batch_index < batch_size; batch_index++){

		//memory reordering and carry adding
		data_im[im_idx] = data_mat[mat_idx]+carry;
		

		//offset both indexers
		im_idx += im_offset;
		mat_idx += mat_offset;
	}
	
  }
}

template <typename Dtype>
void cu_mat2im_c_gpu(const Dtype* data_mat,
    const int mat_height, const int mat_width, 
	Dtype* data_im,
	Dtype beta,
	Dtype* data_carry,
	const int batch_size) {
  // CUDA_CHECK(cudaMemset(data_im, 0,
  //            sizeof(Dtype) * height * width * channels));
  int num_kernels = mat_height * mat_width;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  // NOLINT_NEXT_LINE(whitespace/operators)
  cu_mat2im_c_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS, 0, Caffe::get_current_cu_stream()>>>(
      num_kernels, data_mat, mat_height, mat_width,
	  data_im,
	  beta,
	  data_carry,
	  batch_size);
  //CUDA_POST_KERNEL_CHECK;
}


// Explicit instantiation
template void cu_mat2im_c_gpu<float>(const float* data_mat,
    const int mat_height, const int mat_width, 
	float* data_im,
	float beta,
	float* data_carry,
	const int batch_size);
template void cu_mat2im_c_gpu<double>(const double* data_mat,
    const int mat_height, const int mat_width, 
	double* data_im,
	double beta,
	double* data_carry,
	const int batch_size);


template <typename Dtype>
__global__ void cu_trans_3D_acc_gpu_kernel(const int n, const Dtype* data_src, Dtype *data_dst,
    const int X, const int Y, const int Z, int mX, int mY, int mZ, Dtype alpha)
{
	CUDA_KERNEL_LOOP(index, n) 
	{
		int temp = index;
		int Y_pos = temp % Y;
		int Z_pos = temp / Y;

		int current_pos = index * X;

		for (int X_pos = 0; X_pos < X; X_pos++, current_pos++)
		{
			int npos = X_pos * mX + Y_pos * mY + Z_pos * mZ;
			data_dst[npos] = data_src[current_pos] + alpha * data_dst[npos];
		}
	}
}

template <typename Dtype>
void cu_permute_3D_acc_gpu(const Dtype *data_src, Dtype *data_dst, const int X, const int Y, const int Z, const DimensionOrder transDirection, Dtype alpha)
{
	int num_kernels = Z * Y;

	//multiplier for X axis, Y axis, Z axis
	int mX, mY, mZ;
	switch (transDirection)
	{
	case XYZtoXYZ:
		mX = 1, mY = X, mZ = X * Y;
		break;
	case XYZtoXZY:
		mX = 1, mY = X * Z, mZ = X;
		break;
	case XYZtoYXZ:
		mX = Y, mY = 1, mZ = X * Y;
		break;
	case XYZtoYZX:
		mX = X * Y, mY = 1, mZ = Y;
		break;
	case XYZtoZXY:
		mX = Z, mY = X * Z, mZ = 1;
		break;
	case XYZtoZYX:
		mX = Y * Z, mY = Z, mZ = 1;
		break;
	}

	cu_trans_3D_acc_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
		CAFFE_CUDA_NUM_THREADS, 0, Caffe::get_current_cu_stream()>>>(
		num_kernels, data_src, data_dst, X, Y, Z, mX, mY, mZ, alpha);
}

template void cu_permute_3D_acc_gpu<float>(const float *data_src, float *data_dst, const int X, const int Y, const int Z, const DimensionOrder transDirection, float alpha);
template void cu_permute_3D_acc_gpu<double>(const double *data_src, double *data_dst, const int X, const int Y, const int Z, const DimensionOrder transDirection, double alpha);


template <typename Dtype>
__global__ void cu_trans_3D_carry_gpu_kernel(const int n, const Dtype* data_src, Dtype *data_dst,
											 const Dtype *data_carry,
    const int X, const int Y, const int Z, int mX, int mY, int mZ, Dtype alpha, Dtype beta)
{
	CUDA_KERNEL_LOOP(index, n) 
	{
		int temp = index;
		int Y_pos = temp % Y;
		int Z_pos = temp / Y;

		int current_pos = index * X;

		for (int X_pos = 0; X_pos < X; X_pos++, current_pos++)
		{
			int npos = X_pos * mX + Y_pos * mY + Z_pos * mZ;
			data_dst[npos] = data_src[current_pos] + alpha * data_dst[npos] + beta * data_carry[npos];
		}
	}
}

template <typename Dtype>
void cu_permute_3D_carry_gpu(const Dtype *data_src, Dtype *data_dst, Dtype * data_carry, const int X, const int Y, const int Z,
							 const DimensionOrder transDirection, Dtype alpha, Dtype beta)
{
	int num_kernels = Z * Y;

	//multiplier for X axis, Y axis, Z axis
	int mX, mY, mZ;
	switch (transDirection)
	{
	case XYZtoXYZ:
		mX = 1, mY = X, mZ = X * Y;
		break;
	case XYZtoXZY:
		mX = 1, mY = X * Z, mZ = X;
		break;
	case XYZtoYXZ:
		mX = Y, mY = 1, mZ = X * Y;
		break;
	case XYZtoYZX:
		mX = X * Y, mY = 1, mZ = Y;
		break;
	case XYZtoZXY:
		mX = Z, mY = X * Z, mZ = 1;
		break;
	case XYZtoZYX:
		mX = Y * Z, mY = Z, mZ = 1;
		break;
	}

	cu_trans_3D_carry_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
		CAFFE_CUDA_NUM_THREADS, 0, Caffe::get_current_cu_stream()>>>(
		num_kernels, data_src, data_dst, data_carry,  X, Y, Z, mX, mY, mZ, alpha, beta);
}

template void cu_permute_3D_carry_gpu<float>(const float *data_src, float *data_dst, float * data_carry, const int X, const int Y, const int Z, const DimensionOrder transDirection, float alpha, float beta);
template void cu_permute_3D_carry_gpu<double>(const double *data_src, double *data_dst, double * data_carry, const int X, const int Y, const int Z, const DimensionOrder transDirection, double alpha, double beta);


}  // namespace caffe
