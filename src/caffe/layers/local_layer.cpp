// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void LocallyConnectedLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 1) << "Locally connected Layer takes a single blob as input.";
  CHECK_EQ(top->size(), 1) << "Locally connected Layer takes a single blob as output.";

  //extract layer configurations
  kernel_size_ = this->layer_param_.convolution_param().kernel_size();
  stride_ = this->layer_param_.convolution_param().stride();
  group_ = this->layer_param_.convolution_param().group(); // not sure whether its good to add group to local layers, but keep it
  pad_ = this->layer_param_.convolution_param().pad();
  mem_group_size = this->layer_param_.convolution_param().mem_group_size();
  LOG(INFO)<< "Grouping im2col at size "<<mem_group_size;

  // extract layer info
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  num_output_ = this->layer_param_.convolution_param().num_output();
  CHECK_GT(num_output_, 0);
  CHECK_EQ(channels_ % group_, 0);
  // The im2col result buffer would only hold one image at a time to avoid - Hold mem_group_size images
  // overly large memory usage.
  int height_out = (height_ + 2 * pad_ - kernel_size_) / stride_ + 1;
  int width_out = (width_ + 2 * pad_ - kernel_size_) / stride_ + 1;
  col_buffer_.Reshape(
      mem_group_size, channels_ * kernel_size_ * kernel_size_, height_out, width_out);
  
  // Set the parameters
  CHECK_EQ(num_output_ % group_, 0)
      << "Number of output should be multiples of group.";
  bias_term_ = this->layer_param_.convolution_param().bias_term();
  // Figure out the dimensions for individual gemms.
  M_ = num_output_ / group_;
  K_ = channels_ * kernel_size_ * kernel_size_ / group_;
  N_ = height_out * width_out;
  (*top)[0]->Reshape(bottom[0]->num(), num_output_, height_out, width_out);
 // bias_buffer_.Reshape(
	//num_output_, 1, M_, N_);
  trans_buffer_.Reshape(
	  mem_group_size,1,num_output_, N_);

  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Intialize the weight
    this->blobs_[0].reset(new Blob<Dtype>(
        num_output_, 1, K_, N_));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.convolution_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, intiialize and fill the bias term
    if (bias_term_) {
      this->blobs_[1].reset(new Blob<Dtype>(1, 1, N_, num_output_));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.convolution_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }

  // Set up the bias filler
  if (bias_term_) 
  {
	  bias_multiplier_.reset(new SyncedMemory(num_ * sizeof(Dtype)));
	  Dtype* bias_multiplier_data =
		  reinterpret_cast<Dtype*>(bias_multiplier_->mutable_cpu_data());
	  for (int i = 0; i < num_; i++) {
		  bias_multiplier_data[i] = Dtype(1.);
	  }

	  row_sumer_.reset(new SyncedMemory(num_*sizeof(Dtype)));
	  Dtype* row_sumer_data = reinterpret_cast<Dtype*>(row_sumer_->mutable_cpu_data());
	  for (int i = 0; i < num_; i++){
		  row_sumer_data[i] = 1.;
	  }
  }

  // Set up the entry points buffer for cublas batched gemm
  weight_entry_.reset(new SyncedMemory(N_*sizeof(Dtype*)));
  col_entry_.reset(new SyncedMemory(N_*sizeof(Dtype*)));

  weight_diff_entry_.reset(new SyncedMemory(N_ * sizeof(Dtype*)));
  col_diff_entry_.reset(new SyncedMemory(N_ * sizeof(Dtype*)));
  
  result_entry_.reset(new SyncedMemory(N_ * sizeof(Dtype*)));

  entry_initialized_ = false;
  bp_entry_initialized_ = false;

}


template <typename Dtype>
Dtype LocallyConnectedLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  NOT_IMPLEMENTED;
  return Dtype(0.);
}

template <typename Dtype>
void LocallyConnectedLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  NOT_IMPLEMENTED;
}

INSTANTIATE_CLASS(LocallyConnectedLayer);

}  // namespace caffe
