// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <vector>
#include <string>
#include <fstream>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"


using std::max;

namespace caffe {

template <typename Dtype>
void AccuracyLayer<Dtype>::SetUp(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 2) << "Accuracy Layer takes two blobs as input.";
  CHECK_EQ(top->size(), 1) << "Accuracy Layer takes 1 output.";
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "The data and label should have the same number.";
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  (*top)[0]->Reshape(1, 2, 1, 1);
}

template <typename Dtype>
Dtype AccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  Dtype accuracy = 0;
  Dtype logprob = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / bottom[0]->num();
  for (int i = 0; i < num; ++i) {
    // Accuracy
    Dtype maxval = -FLT_MAX;
    int max_id = 0;
    for (int j = 0; j < dim; ++j) {
      if (bottom_data[i * dim + j] > maxval) {
        maxval = bottom_data[i * dim + j];
        max_id = j;
      }
    }
    if (max_id == static_cast<int>(bottom_label[i])) {
      ++accuracy;
    }
    Dtype prob = max(bottom_data[i * dim + static_cast<int>(bottom_label[i])],
                     Dtype(kLOG_THRESHOLD));
    logprob -= log(prob);
  }
  // LOG(INFO) << "Accuracy: " << accuracy;
  (*top)[0]->mutable_cpu_data()[0] = accuracy / num;
  (*top)[0]->mutable_cpu_data()[1] = logprob / num;
  // Accuracy layer should not be used as a loss function.
  return Dtype(0);
}

INSTANTIATE_CLASS(AccuracyLayer);

template <typename Dtype>
void MultiViewAccuracyLayer<Dtype>::SetUp(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 2) << "Accuracy Layer takes two blobs as input.";
  CHECK_EQ(top->size(), 1) << "Accuracy Layer takes 1 output.";
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "The data and label should have the same number.";
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  (*top)[0]->Reshape(1, 2, 1, 1);
}

template <typename Dtype>
Dtype MultiViewAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  Dtype accuracy = 0;
  Dtype logprob = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  int view_num = bottom[0]->num();
  int num = view_num / 10;
  int dim = bottom[0]->count() / bottom[0]->num();
  for (int i = 0; i < num; ++i) {
    // sum up scores of 10 views
    vector<Dtype> score(dim);
    for (int v = 0; v < 10; v++) {
      for (int j = 0; j < dim; j++) {
        score[j] += bottom_data[(i * 10 + v) * dim + j];
      }
    }
    Dtype maxval = -FLT_MAX;
    int max_id = 0;
    for (int j = 0; j < dim; j++) {
      score[j] /= 10;
      if (score[j] > maxval) {
        maxval = score[j];
        max_id = j;
      }
    }
    if (max_id == static_cast<int>(bottom_label[i * 10])) {
      ++accuracy;
    }
    Dtype prob = max(score[static_cast<int>(bottom_label[i * 10])],
                     Dtype(kLOG_THRESHOLD));
    logprob -= log(prob);
  }
  // LOG(INFO) << "Accuracy: " << accuracy;
  (*top)[0]->mutable_cpu_data()[0] = accuracy / num;
  (*top)[0]->mutable_cpu_data()[1] = logprob / num;
  // Accuracy layer should not be used as a loss function.
  return Dtype(0);
}

template <typename  Dtype>
void TargetCodingAccuracyLayer<Dtype>::SetUp(
const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  // call base-class function first
  AccuracyLayer<Dtype>::SetUp(bottom, top);

  LOG(INFO) << "START LOAD LABEL_CODE_TABLE";
  CHECK_EQ(bottom[0]->channels(), label_length_)
    << " predict output channels shoud be same as label code " << std::endl;

  std::ifstream infile(code_file_.c_str());
  CHECK(infile.good()) << "Failed to open target code file: " << code_file_ << std::endl;

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
Dtype TargetCodingAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  Dtype accuracy = 0;
  Dtype logprob = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / bottom[0]->num();
  Dtype loss = 0;
  for (int i = 0; i < num; ++i) {
      // Accuracy
      int min_id = 0;
      Dtype min_value = FLT_MAX;
      int gt_id = static_cast<int>(bottom_label[i]);
      // tranverse the code book to find the nearest code
      for (int label_code_ind = 0; label_code_ind < label_class_num_; ++label_code_ind){
          Dtype diff_pred_code_ind = 0;
          for (int j = 0; j < dim; ++j) {
              Dtype tmp = bottom_data[i * dim + j] - label_code_table_[label_code_ind][j];
              //diff_pred_code_ind = diff_pred_code_ind + abs(bottom_data[i * dim + j] - label_code_table_[label_code_ind][j]);
              diff_pred_code_ind += tmp * tmp;
          }
          if(diff_pred_code_ind < min_value){
              min_id = label_code_ind;
              min_value = diff_pred_code_ind;
          }
          if (label_code_ind == gt_id)
            loss += diff_pred_code_ind;
      }
      if (min_id == gt_id){
          ++accuracy;
      }
  }
  // LOG(INFO) << "Accuracy: " << accuracy;
  (*top)[0]->mutable_cpu_data()[0] = accuracy / num;
  //LOG(INFO) << "Accuracy: " << accuracy << " num:"<< num;
  (*top)[0]->mutable_cpu_data()[1] = sqrt(loss / num / label_length_);
  // Accuracy layer should not be used as a loss function.
  return Dtype(0);
}

INSTANTIATE_CLASS(MultiViewAccuracyLayer);

INSTANTIATE_CLASS(TargetCodingAccuracyLayer);

INSTANTIATE_CLASS(TargetCoding1023_569AccuracyLayer);

INSTANTIATE_CLASS(TargetCoding255_200AccuracyLayer);

INSTANTIATE_CLASS(TargetCoding1023_1000AccuracyLayer);

template <typename Dtype>
Dtype TargetCodingBinaryAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  Dtype accuracy = 0;
  Dtype hammingdist = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / bottom[0]->num();
  Dtype* binary_bottom_data = new Dtype[this->label_length_];
  for (int i = 0; i < num; ++i) {
      // Accuracy
      int min_id = 0;
      Dtype min_value = FLT_MAX;
      int gt_id = static_cast<int>(bottom_label[i]);
      // tranverse the code book to find the nearest code
      for (int j = 0; j < dim; j++) {
        binary_bottom_data[j] = 0;
        if (bottom_data[i * dim + j] > 0.5)
          binary_bottom_data[j] = 1;
      }
      for (int label_code_ind = 0; label_code_ind < this->label_class_num_; ++label_code_ind){
          Dtype diff_pred_code_ind = 0;
          for (int j = 0; j < dim; ++j) {
            diff_pred_code_ind = diff_pred_code_ind + abs(binary_bottom_data[j] - this->label_code_table_[label_code_ind][j]);
          }
          if(diff_pred_code_ind < min_value){
              min_id = label_code_ind;
              min_value = diff_pred_code_ind;
          }
          if (label_code_ind == gt_id)
            hammingdist += diff_pred_code_ind;
      }
      if (min_id == gt_id){
          ++accuracy;
      }
  }
  // LOG(INFO) << "Accuracy: " << accuracy;
  (*top)[0]->mutable_cpu_data()[0] = accuracy / num;
  //LOG(INFO) << "Accuracy: " << accuracy << " num:"<< num;
  (*top)[0]->mutable_cpu_data()[1] = sqrt(hammingdist / num / this->label_length_);
  // Accuracy layer should not be used as a loss function.
  return Dtype(0);
}

INSTANTIATE_CLASS(TargetCodingBinaryAccuracyLayer);

INSTANTIATE_CLASS(TargetCodingBinary1023_1000AccuracyLayer);

// INSTANTIATE_CLASS(TargetCodingBinary1023_1000AccuracyLayer);
}  // namespace caffe
