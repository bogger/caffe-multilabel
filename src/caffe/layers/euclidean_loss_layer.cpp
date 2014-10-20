// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"

using std::max;

namespace caffe {

template <typename Dtype>
void L2LossLayer<Dtype>::FurtherSetUp(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
  diff_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
Dtype L2LossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  int count = bottom[0]->count();
  const Dtype *data0 = bottom[0]->cpu_data();
  const Dtype *data1 = bottom[1]->cpu_data();
  for (int i = 0; i < 4; i++) {
    std::cout << data0[i] << ", " << data1[i] << std::endl;
  }
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());
  Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
  Dtype loss = dot / bottom[0]->num() / Dtype(2);
  return loss;
}

template <typename Dtype>
void L2LossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  caffe_cpu_axpby(
      (*bottom)[0]->count(),
      Dtype(1) / (*bottom)[0]->num(),
      diff_.cpu_data(),
      Dtype(0),
      (*bottom)[0]->mutable_cpu_diff());
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::FurtherSetUp(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom[0]->channels(), label_length_);
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
  diff_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());

  LOG(INFO)<<"START LOAD LABEL_CODE_TABLE";
  // label_length_=255;
  CHECK_EQ(bottom[0]->channels(), label_length_)
      << " predict output channels shoud be same as label code "<<std::endl;

  // label_class_num_=201;
  LOG(INFO)<<"Start load label_code_table file";
  std::ifstream infile(code_file_.c_str());
  CHECK(infile.good()) << "Failed to open target code file: " << code_file_ << std::endl;

  int label_class_num_ = label_length_;
  for(int i = 0; i < label_class_num_; ++i){
      // LOG(INFO)<< "START LOAD LABEL_CODE INDEX: "<<i<<"-"<<label_class_num_;
      vector<Dtype> label_code;
      for(int j = 0; j < label_length_; ++j){
          //LOG(INFO) << "start load label_code bit index: "<< j<<"-"<< label_length_;
          Dtype label_bit;
          infile >> label_bit;
          label_code.push_back(label_bit);
      }
      label_code_table_.push_back(label_code);
  }
  LOG(INFO)<<"END LOADING LABEL_CODE_TABLE";
  infile.close();
}

template <typename Dtype>
Dtype EuclideanLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  int count = bottom[0]->count();

  Blob<Dtype> gt_target_code(bottom[0]->num(),bottom[0]->channels(),bottom[0]->height(),bottom[0]->width());
  CHECK_EQ(bottom[0]->height(),1);
  CHECK_EQ(bottom[0]->width(),1);
  for(int i=0;i<bottom[0]->num();++i){
      for(int j=0;j<bottom[0]->channels();++j){
          *(gt_target_code.mutable_cpu_data() + gt_target_code.offset(i,j))
            = label_code_table_[bottom[1]->cpu_data()[i]][j];
      }
  }
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      gt_target_code.cpu_data(),
      diff_.mutable_cpu_data());

  // for (int i = 0; i < 10; i++) {
  //   std::cout << std::setprecision(3) << bottom[0]->cpu_data()[i] << ","
  //     << gt_target_code.cpu_data()[i] << "  ";
  // }
  // std::cout << std::endl;
  Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
  Dtype loss = sqrt(dot / bottom[0]->num() / label_length_);
  //LOG(INFO)<< dot << " " << bottom[0]->num() << " " << bottom[0]->data_at(0, 0, 0, 0);
  return loss;
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  caffe_cpu_axpby(
      (*bottom)[0]->count(),              // count
      Dtype(1) / (*bottom)[0]->num(),     // alpha
      diff_.cpu_data(),                   // a
      Dtype(0),                           // beta
      (*bottom)[0]->mutable_cpu_diff());  // b
}

template <typename Dtype>
void TargetCodingHingeLossLayer<Dtype>::FurtherSetUp(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom[0]->channels(), label_length_);
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
  gt_label_code_.Reshape(bottom[0]->num(), bottom[0]->channels(),
            bottom[0]->height(), bottom[0]->width());

  LOG(INFO)<<"START LOAD LABEL_CODE_TABLE";
  // label_length_=255;
  CHECK_EQ(bottom[0]->channels(), label_length_)
      << " predict output channels shoud be same as label code "<<std::endl;

  std::ifstream infile(code_file_.c_str());
  CHECK(infile.good()) << "Failed to open target code file: " << code_file_ << std::endl;

  int label_class_num_ = label_length_;
  for(int i = 0; i < label_class_num_; ++i){
      vector<Dtype> label_code;
      for(int j = 0; j < label_length_; ++j){
          Dtype label_bit;
          infile >> label_bit;
          label_code.push_back(label_bit);
      }
      label_code_table_.push_back(label_code);
  }
  LOG(INFO)<<"END LOADING LABEL_CODE_TABLE";
  infile.close();
}

template <typename Dtype>
Dtype TargetCodingHingeLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {

  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
  int num = bottom[0]->num();
  int count = bottom[0]->count();
  int dim = count / num;
  CHECK_EQ(dim, label_length_);
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* label = bottom[1]->cpu_data();
  Dtype* label_code = gt_label_code_.mutable_cpu_data();

  for(int i = 0; i < bottom[0]->num(); ++i){
      for(int j = 0; j < bottom[0]->channels(); ++j){
          *(label_code + gt_label_code_.offset(i,j))
            = label_code_table_[label[i]][j];
      }
  }

  caffe_copy(count, bottom_data, bottom_diff);
  caffe_add_scalar(count, Dtype(-0.5), bottom_diff); // b-0.5
  caffe_scal(count, Dtype(-2), label_code);
  caffe_add_scalar(count, Dtype(1), label_code);  // 1-2*gt
  caffe_mul(count, bottom_diff, label_code, bottom_diff);
  for (int i = 0; i < num; i++) {
    for (int j = 0; j < dim; j++) {
      bottom_diff[i * dim + j] = max(Dtype(0), margin_ + bottom_diff[i * dim + j]);
    }
  }

  //Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
  Dtype loss = caffe_cpu_asum(count, bottom_diff) / num / label_length_; // hinge loss
  loss = Dtype(0.5) - margin_ + loss;  // loss relative to the groudtruth
  //LOG(INFO)<< dot << " " << bottom[0]->num() << " " << bottom[0]->data_at(0, 0, 0, 0);
  return loss;
}

template <typename Dtype>
void TargetCodingHingeLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  Dtype *bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  const Dtype *label_code = gt_label_code_.mutable_cpu_data();
  int num = (*bottom)[0]->num();
  int count = (*bottom)[0]->count();
  const Dtype* bottom_data = (*bottom)[0]->cpu_data();

  caffe_cpu_sign(count, bottom_diff, bottom_diff);
  caffe_mul(count, bottom_diff, label_code, bottom_diff);
  caffe_scal(count, Dtype(1. / num), bottom_diff);

  // for (int i = 512; i < 512 + 5; i++) {
  //   std::cout << std::setprecision(3) << bottom_data[i] << ","
  //     << label_code[i] << "," << bottom_diff[i] << "   ";
  // }
  // std::cout << std::endl;
}

INSTANTIATE_CLASS(L2LossLayer);

INSTANTIATE_CLASS(EuclideanLossLayer);


INSTANTIATE_CLASS(Euclidean1023LossLayer);


INSTANTIATE_CLASS(Euclidean255LossLayer);

INSTANTIATE_CLASS(TargetCodingHingeLossLayer);

INSTANTIATE_CLASS(TargetCoding1023HingeLossLayer);

}  // namespace caffe
