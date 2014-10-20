// Copyright 2014 BVLC and contributors.

#ifndef CAFFE_LOSS_LAYERS_HPP_
#define CAFFE_LOSS_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "leveldb/db.h"
#include "pthread.h"
#include "boost/scoped_ptr.hpp"
#include "hdf5.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/neuron_layers.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

const float kLOG_THRESHOLD = 1e-20;

/* LossLayer
  Takes two inputs of same num (a and b), and has no output.
  The gradient is propagated to a.
*/
template <typename Dtype>
class LossLayer : public Layer<Dtype> {
 public:
  explicit LossLayer(const LayerParameter& param)
     : Layer<Dtype>(param) {}
  virtual void SetUp(
      const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top);
  virtual void FurtherSetUp(
      const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {}
};

/* SigmoidCrossEntropyLossLayer
*/
template <typename Dtype>
class SigmoidCrossEntropyLossLayer : public LossLayer<Dtype> {
 public:
  explicit SigmoidCrossEntropyLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param),
          sigmoid_layer_(new SigmoidLayer<Dtype>(param)),
          sigmoid_output_(new Blob<Dtype>()) {}
  virtual void FurtherSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);

  shared_ptr<SigmoidLayer<Dtype> > sigmoid_layer_;
  // sigmoid_output stores the output of the sigmoid layer.
  shared_ptr<Blob<Dtype> > sigmoid_output_;
  // Vector holders to call the underlying sigmoid layer forward and backward.
  vector<Blob<Dtype>*> sigmoid_bottom_vec_;
  vector<Blob<Dtype>*> sigmoid_top_vec_;
};

/* EuclideanLossLayer
  Compute the L_2 distance between the two inputs.

  loss = (1/2 \sum_i (a_i - b_i)^2)
  a' = 1/I (a - b)
*/
template <typename Dtype>
class L2LossLayer : public LossLayer<Dtype> {
 public:
  explicit L2LossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param), diff_() {}
  virtual void FurtherSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);

  Blob<Dtype> diff_;
};

template <typename Dtype>
class EuclideanLossLayer : public LossLayer<Dtype> {
 public:
  explicit EuclideanLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param), diff_() {}
  explicit EuclideanLossLayer(const LayerParameter& param, const int label_length, const std::string code_file)
      : LossLayer<Dtype>(param), diff_(), label_length_(label_length), code_file_(code_file) {}
  virtual void FurtherSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);

  Blob<Dtype> diff_;
  vector< vector<Dtype> > label_code_table_;
  int label_length_;
  std::string code_file_;
};

template <typename Dtype>
class Euclidean1023LossLayer : public EuclideanLossLayer<Dtype> {
 public:
  explicit Euclidean1023LossLayer(const LayerParameter& param)
      : EuclideanLossLayer<Dtype>(param, 1023,"/home/common/caffe-master/src/caffe/layers/singer_difference_1023.txt") {}
};

template <typename Dtype>
class Euclidean255LossLayer : public EuclideanLossLayer<Dtype> {
 public:
  explicit Euclidean255LossLayer(const LayerParameter& param)
      : EuclideanLossLayer<Dtype>(param, 255, "/home/common/caffe-master/src/caffe/layers/singer_difference_255.txt") {}
};


template <typename Dtype>
class TargetCodingHingeLossLayer : public LossLayer<Dtype> {
public:
  explicit TargetCodingHingeLossLayer(const LayerParameter& param, const int label_length,
                        const float margin, const std::string code_file)
      : LossLayer<Dtype>(param), margin_(margin), gt_label_code_(), label_length_(label_length), code_file_(code_file) {}
  virtual void FurtherSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);

  Blob<Dtype> gt_label_code_;
  vector< vector<Dtype> > label_code_table_;
  int label_length_;
  float margin_;
  std::string code_file_;
};

template <typename Dtype>
class TargetCoding1023HingeLossLayer : public TargetCodingHingeLossLayer<Dtype> {
 public:
  explicit TargetCoding1023HingeLossLayer(const LayerParameter& param)
      : TargetCodingHingeLossLayer<Dtype>(param, 1023, 0.1f, "/home/common/caffe-master/src/caffe/layers/singer_difference_1023.txt") {}
};

/* InfogainLossLayer
*/
template <typename Dtype>
class InfogainLossLayer : public LossLayer<Dtype> {
 public:
  explicit InfogainLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param), infogain_() {}
  virtual void FurtherSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);

  Blob<Dtype> infogain_;
};

/* HingeLossLayer
*/
template <typename Dtype>
class HingeLossLayer : public LossLayer<Dtype> {
 public:
  explicit HingeLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
};

template <typename Dtype>
class HingeLoss4DetLayer : public LossLayer<Dtype> {
 public:
  explicit HingeLoss4DetLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
};

/* MultinomialLogisticLossLayer
*/
template <typename Dtype>
class MultinomialLogisticLossLayer : public LossLayer<Dtype> {
 public:
  explicit MultinomialLogisticLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void FurtherSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
};

/* AccuracyLayer
  Note: not an actual loss layer! Does not implement backwards step.
  Computes the accuracy and logprob of a with respect to b.
*/
template <typename Dtype>
class AccuracyLayer : public Layer<Dtype> {
 public:
  explicit AccuracyLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
    NOT_IMPLEMENTED;
  }
};


template <typename Dtype>
class MultiViewAccuracyLayer : public Layer<Dtype> {
 public:
  explicit MultiViewAccuracyLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
    NOT_IMPLEMENTED;
  }
};
/* Also see
- SoftmaxWithLossLayer in vision_layers.hpp
*/


template <typename Dtype>
class TargetCodingAccuracyLayer : public AccuracyLayer<Dtype> {
 public:
  explicit TargetCodingAccuracyLayer(const LayerParameter& param, const int label_len, const int class_num, const std::string code_file)
      : AccuracyLayer<Dtype>(param) {
        label_length_ = label_len;
        label_class_num_ = class_num;
        code_file_ = code_file;
      }
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
    NOT_IMPLEMENTED;
  }
  vector< vector<Dtype > > label_code_table_;
  int label_length_;
  int label_class_num_;
  std::string  code_file_;
};



template <typename Dtype>
class TargetCoding1023_569AccuracyLayer : public TargetCodingAccuracyLayer<Dtype> {
 public:
  explicit TargetCoding1023_569AccuracyLayer(const LayerParameter& param)
      : TargetCodingAccuracyLayer<Dtype>(param, 1023, 569,
                                  "/home/common/caffe-master/src/caffe/layers/singer_difference_1023.txt"){}
};

template <typename Dtype>
class TargetCoding255_200AccuracyLayer : public TargetCodingAccuracyLayer<Dtype> {
 public:
  explicit TargetCoding255_200AccuracyLayer(const LayerParameter& param)
      : TargetCodingAccuracyLayer<Dtype>(param, 255, 200,
                                  "/home/common/caffe-master/src/caffe/layers/singer_difference_255.txt"){}
};

template <typename Dtype>
class TargetCoding1023_1000AccuracyLayer : public TargetCodingAccuracyLayer<Dtype> {
 public:
  explicit TargetCoding1023_1000AccuracyLayer(const LayerParameter& param)
      : TargetCodingAccuracyLayer<Dtype>(param, 1023, 1000,
                                  "/home/common/caffe-master/src/caffe/layers/singer_difference_1023.txt"){}
};

template <typename Dtype>
class TargetCodingBinaryAccuracyLayer : public TargetCodingAccuracyLayer<Dtype> {
 public:
  explicit TargetCodingBinaryAccuracyLayer(const LayerParameter& param, const int label_len, const int class_num, const std::string code_file)
      : TargetCodingAccuracyLayer<Dtype>(param, label_len, class_num, code_file) {}
  protected:
    virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
};

template <typename Dtype>
class TargetCodingBinary1023_1000AccuracyLayer : public TargetCodingBinaryAccuracyLayer<Dtype> {
 public:
  explicit TargetCodingBinary1023_1000AccuracyLayer(const LayerParameter& param)
      : TargetCodingBinaryAccuracyLayer<Dtype>(param, 1023, 1000,
                                  "/home/common/caffe-master/src/caffe/layers/singer_difference_1023.txt"){}
};

} // namespace caffe
#endif  // CAFFE_LOSS_LAYERS_HPP_
