// Copyright 2014 BVLC and contributors.

#include <stdint.h>
#include <leveldb/db.h>
#include <pthread.h>

#include <string>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"

using std::string;

namespace caffe {

template <typename Dtype>
Dtype DataLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // First, join the thread
  JoinPrefetchThread();
  // Copy the data
  CUDA_CHECK(cudaMemcpy((*top)[0]->mutable_gpu_data(),
      prefetch_data_->cpu_data(), sizeof(Dtype) * prefetch_data_->count(),
      cudaMemcpyHostToDevice));
  if (output_labels_) {
    

    int batch_size = this->layer_param_.data_param().batch_size();
    for (int i=0; i<prefetch_label_->channels(); i++) {

      CUDA_CHECK(cudaMemcpy((*top)[i+1]->mutable_gpu_data(),
        prefetch_label_->cpu_data() + i * batch_size, sizeof(Dtype) * batch_size,
        cudaMemcpyHostToDevice));
    }
  }
  // Start a new prefetch thread
  CreatePrefetchThread();
  return Dtype(0.);
}

INSTANTIATE_CLASS(DataLayer);

}  // namespace caffe
