// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"

using std::max;

namespace caffe {

/* note that in the window_data_layer, the labels are assigned this way:
    bg      --> 0
    cls 1   --> 1
    cls 2   --> 2
    ...
    cls 200 --> 200
*/
template <typename Dtype>
Dtype HingeLoss4DetLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* label = bottom[1]->cpu_data();
  int num = bottom[0]->num();
  int count = bottom[0]->count();
  int dim = count / num;

  caffe_copy(count, bottom_data, bottom_diff);
  for (int i = 0; i < num; ++i) {
    if (label[i] > 0)
      bottom_diff[i * dim + static_cast<int>(label[i] - 1)] *= -1;
  }
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < dim; ++j) {
      bottom_diff[i * dim + j] = max(Dtype(0), 1 + bottom_diff[i * dim + j]);
    }
  }
  return caffe_cpu_asum(count, bottom_diff) / num;
}

template <typename Dtype>
void HingeLoss4DetLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  const Dtype* label = (*bottom)[1]->cpu_data();
  int num = (*bottom)[0]->num();
  int count = (*bottom)[0]->count();
  int dim = count / num;

  caffe_cpu_sign(count, bottom_diff, bottom_diff);
  for (int i = 0; i < num; ++i) {
    if (label[i] > 0)
      bottom_diff[i * dim + static_cast<int>(label[i] - 1)] *= -1;
  }
  caffe_scal(count, Dtype(1. / num), bottom_diff);
}

INSTANTIATE_CLASS(HingeLoss4DetLayer);

}  // namespace caffe
