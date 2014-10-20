// Copyright 2014 BVLC and contributors.

#ifndef _CAFFE_UTIL_IM2COL_HPP_
#define _CAFFE_UTIL_IM2COL_HPP_

namespace caffe {

template <typename Dtype>
void im2col_cpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int ksize, const int pad,
    const int stride, Dtype* data_col);

template <typename Dtype>
void col2im_cpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int psize, const int pad,
    const int stride, Dtype* data_im);

template <typename Dtype>
void im2col_gpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int ksize, const int pad,
    const int stride, Dtype* data_col);

template <typename Dtype>
void col2im_gpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int psize, const int pad,
    const int stride, Dtype* data_im);

template <typename Dtype>
void bu_im2col_gpu(const Dtype* data_im, const int channels,
				   const int height, const int width, const int ksize, const int pad,
				   const int stride, Dtype* data_col, const int batch_size);

template <typename Dtype>
void bu_im2col_gpu(const Dtype* data_im, const int channels,
				   const int height, const int width, const int ksize, const int pad,
				   const int stride, Dtype* data_col, const int batch_size, const bool trans);

template <typename Dtype>
void bu_im2col_gpu_rot(const Dtype* data_im, const int channels,
				   const int height, const int width, const int ksize, const int pad,
				   const int stride, Dtype* data_col, const int batch_size);

template <typename Dtype>
void bu_col2im_gpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int psize, const int pad,
    const int stride, Dtype* data_im,
	const int batch_size);

template <typename Dtype>
void bu_col2im_gpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int psize, const int pad,
    const int stride, Dtype* data_im,
	const int batch_size, bool trans);

template <typename Dtype>
void bu_col2im_gpu_rot(const Dtype* data_col, const int channels,
    const int height, const int width, const int ksize, const int pad,
    const int stride, Dtype* data_im,
	const int batch_size);

template <typename Dtype>
void cu_im2mat_gpu(const Dtype* data_im, const int channels,
    const int height, const int width, 
	Dtype* data_mat,
	const int batch_size);

template <typename Dtype>
void cu_mat2im_c_gpu(const Dtype* data_mat,
    const int mat_height, const int mat_width, 
	Dtype* data_im,
	Dtype beta,
	Dtype* data_carry,
	const int batch_size);

enum DimensionOrder
{
	XYZtoXYZ = 0,
	XYZtoXZY,
	XYZtoYXZ,
	XYZtoYZX,
	XYZtoZXY,
	XYZtoZYX
};

template <typename Dtype>
void cu_permute_3D_acc_gpu(const Dtype *data_src, Dtype *data_dst, const int X, const int Y, const int Z, const DimensionOrder transDirection, Dtype alpha = (Dtype)0.0);
template <typename Dtype>
void cu_permute_3D_carry_gpu(const Dtype *data_src, Dtype *data_dst, Dtype * data_carry, const int X, const int Y, const int Z,
							 const DimensionOrder transDirection, Dtype alpha=(Dtype)0.0, Dtype beta=(Dtype)1.0);


} // namespace caffe

#endif  // CAFFE_UTIL_IM2COL_HPP_
