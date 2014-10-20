// Copyright 2014 BVLC and contributors.

#ifndef CAFFE_DATA_LAYERS_HPP_
#define CAFFE_DATA_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "leveldb/db.h"
#include "pthread.h"
#include "hdf5.h"
#include "boost/scoped_ptr.hpp"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

#define HDF5_DATA_DATASET_NAME "data"
#define HDF5_DATA_LABEL_NAME "label"

template <typename Dtype>
class HDF5OutputLayer : public Layer<Dtype> {
 public:
  explicit HDF5OutputLayer(const LayerParameter& param);
  virtual ~HDF5OutputLayer();
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  inline std::string file_name() const { return file_name_; }

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void SaveBlobs();

  std::string file_name_;
  hid_t file_id_;
  Blob<Dtype> data_blob_;
  Blob<Dtype> label_blob_;
};


template <typename Dtype>
class HDF5DataLayer : public Layer<Dtype> {
 public:
  explicit HDF5DataLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual ~HDF5DataLayer();
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
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
  virtual void LoadHDF5FileData(const char* filename);

  std::vector<std::string> hdf_filenames_;
  unsigned int num_files_;
  unsigned int current_file_;
  hsize_t current_row_;
  Blob<Dtype> data_blob_;
  Blob<Dtype> label_blob_;
};

// TODO: DataLayer, ImageDataLayer, and WindowDataLayer all have the
// same basic structure and a lot of duplicated code.

// This function is used to create a pthread that prefetches the data.
template <typename Dtype>
void* DataLayerPrefetch(void* layer_pointer);

template <typename Dtype>
void* DataPool5LayerPrefetch(void *layer_poiter);

template <typename Dtype>
void* MultiViewDataLayerPrefetch(void* layer_pointer);

template <typename Dtype>
void* CompactDataLayerPrefetch(void* layer_pointer);

template <typename Dtype>
void* CompactDataBBoxLayerPrefetch(void* layer_pointer);

template <typename Dtype>
void* MultiViewCompactDataLayerPrefetch(void* layer_pointer);

template <typename Dtype>
class DataLayer : public Layer<Dtype> {
  // The function used to perform prefetching.
  friend void* DataLayerPrefetch<Dtype>(void* layer_pointer);

 public:
  explicit DataLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual ~DataLayer();
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) { return; }
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) { return; }

  virtual void CreatePrefetchThread();
  virtual void JoinPrefetchThread();
  virtual unsigned int PrefetchRand();

  shared_ptr<Caffe::RNG> prefetch_rng_;
  shared_ptr<leveldb::DB> db_;
  shared_ptr<leveldb::Iterator> iter_;
  int datum_channels_;
  int datum_height_;
  int datum_width_;
  int datum_size_;
  pthread_t thread_;
  shared_ptr<Blob<Dtype> > prefetch_data_;
  shared_ptr<Blob<Dtype> > prefetch_label_;
  Blob<Dtype> data_mean_;
  bool output_labels_;
  Caffe::Phase phase_;
};

template <typename Dtype>
class DataPool5Layer : public DataLayer<Dtype> {
  friend void* DataPool5LayerPrefetch<Dtype>(void* layer_pointer);

 public:
  explicit DataPool5Layer(const LayerParameter& param)
      : DataLayer<Dtype>(param) {}
  protected:
    virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
    virtual void CreatePrefetchThread();
    int fg_counter_;
    int bg_counter_;
    vector<std::string> fg_keys_;
    vector<std::string> bg_keys_;
    vector<float> fg_overlaps_;
    vector<float> bg_overlaps_;
    shared_ptr<leveldb::DB> fg_db_;
    shared_ptr<leveldb::Iterator> fg_iter_;
    shared_ptr<leveldb::DB> bg_db_;
    shared_ptr<leveldb::Iterator> bg_iter_;
};

template <typename Dtype>
class MultiViewDataLayer : public DataLayer<Dtype> {
  friend void* MultiViewDataLayerPrefetch<Dtype>(void* layer_pointer);

 public:
  explicit MultiViewDataLayer(const LayerParameter& param)
      : DataLayer<Dtype>(param) {}

 protected:
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void CreatePrefetchThread();
};

// inherit DataLayer
template <typename Dtype>
class CompactDataLayer : public DataLayer<Dtype> {
  friend void* CompactDataLayerPrefetch<Dtype>(void* layer_pointer);

 public:
  explicit CompactDataLayer(const LayerParameter& param)
      : DataLayer<Dtype>(param) {}

 protected:
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void CreatePrefetchThread();
};

template <typename Dtype>
class CompactDataBBoxLayer : public DataLayer<Dtype> {
  friend void* CompactDataBBoxLayerPrefetch<Dtype>(void* layer_pointer);

 public:
  explicit CompactDataBBoxLayer(const LayerParameter& param)
      : DataLayer<Dtype>(param) {}
 protected:
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void CreatePrefetchThread();
};

template <typename Dtype>
class MultiViewCompactDataLayer : public DataLayer<Dtype> {
  friend void* MultiViewCompactDataLayerPrefetch<Dtype>(void* layer_pointer);

 public:
  explicit MultiViewCompactDataLayer(const LayerParameter& param)
      : DataLayer<Dtype>(param) {}

 protected:
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void CreatePrefetchThread();
};

// This function is used to create a pthread that prefetches the data.
template <typename Dtype>
void* ImageDataLayerPrefetch(void* layer_pointer);

template <typename Dtype>
class ImageDataLayer : public Layer<Dtype> {
  // The function used to perform prefetching.
  friend void* ImageDataLayerPrefetch<Dtype>(void* layer_pointer);

 public:
  explicit ImageDataLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual ~ImageDataLayer();
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) { return; }
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) { return; }

  virtual void ShuffleImages();

  virtual void CreatePrefetchThread();
  virtual void JoinPrefetchThread();
  virtual unsigned int PrefetchRand();

  shared_ptr<Caffe::RNG> prefetch_rng_;
  vector<std::pair<std::string, int> > lines_;
  int lines_id_;
  int datum_channels_;
  int datum_height_;
  int datum_width_;
  int datum_size_;
  pthread_t thread_;
  shared_ptr<Blob<Dtype> > prefetch_data_;
  shared_ptr<Blob<Dtype> > prefetch_label_;
  Blob<Dtype> data_mean_;
  Caffe::Phase phase_;
};


// This function is used to create a pthread that prefetches the window data.
template <typename Dtype>
void* WindowDataLayerPrefetch(void* layer_pointer);

template <typename Dtype>
void* WindowDataNoBgLayerPrefetch(void* layer_pointer);

template <typename Dtype>
class WindowDataLayer : public Layer<Dtype> {
  // The function used to perform prefetching.
  friend void* WindowDataLayerPrefetch<Dtype>(void* layer_pointer);
  friend void* WindowDataNoBgLayerPrefetch<Dtype>(void* layer_pointer);

 public:
  explicit WindowDataLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual ~WindowDataLayer();
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) { return; }
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) { return; }

  virtual void CreatePrefetchThread();
  virtual void JoinPrefetchThread();
  virtual unsigned int PrefetchRand();

  shared_ptr<Caffe::RNG> prefetch_rng_;
  pthread_t thread_;
  shared_ptr<Blob<Dtype> > prefetch_data_;
  shared_ptr<Blob<Dtype> > prefetch_label_;
  Blob<Dtype> data_mean_;
  vector<std::pair<std::string, vector<int> > > image_database_;
  enum WindowField { IMAGE_INDEX, LABEL, OVERLAP, X1, Y1, X2, Y2, NUM };
  vector<vector<float> > fg_windows_;
  vector<vector<float> > bg_windows_;
  vector<vector<float> > all_windows_;
  int cnt_;
};


// -----------added by arthur-------------

template <typename Dtype>
class WindowDataNoBgLayer : public WindowDataLayer<Dtype> {
 public:
  explicit WindowDataNoBgLayer(const LayerParameter& param)
      : WindowDataLayer<Dtype>(param) {}
 protected:
  virtual void CreatePrefetchThread();
};

// This function is used to create a pthread that prefetches the window data.
template <typename Dtype>
void* WindowDataFeatExtractLayerPrefetch(void* layer_pointer);
template <typename Dtype>
void* WindowDataFeatExtractLayerPrefetch2(void* layer_pointer);
template <typename Dtype>
void* PrefetchHelper(void* arg);
template <typename Dtype>
void* WindowDataFeatExtractLayerPrefetch3Helper(void* layer_pointer);
template <typename Dtype>
void* WindowDataFeatExtractLayerPrefetch3(void* layer_pointer);

// -----------added by arthur-------------
// This class is used to extract window data for feature extraction,
// 		and do not perform permutation on windows.
template <typename Dtype>
class WindowDataFeatExtractLayer : public Layer<Dtype> {
  // The function used to perform prefetching.
  friend void* WindowDataFeatExtractLayerPrefetch<Dtype>(void* layer_pointer);
  friend void* WindowDataFeatExtractLayerPrefetch2<Dtype>(void* layer_pointer);
  friend void* PrefetchHelper<Dtype>(void* arg);
  friend void* WindowDataFeatExtractLayerPrefetch3Helper<Dtype>(void* layer_pointer);
  friend void* WindowDataFeatExtractLayerPrefetch3<Dtype>(void* layer_pointer);

 public:
  explicit WindowDataFeatExtractLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual ~WindowDataFeatExtractLayer();
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

// ----------no need to do back-propagation-------------
 virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
     const bool propagate_down, vector<Blob<Dtype>*>* bottom) { return; }
 virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) { return; }

  virtual void CreatePrefetchThread();
  virtual void JoinPrefetchThread();
  virtual unsigned int PrefetchRand();

 // --------------------- no need to do random permutation --------------------
//  shared_ptr<Caffe::RNG> prefetch_rng_;
  pthread_t thread_, thread1, thread2;
  shared_ptr<Blob<Dtype> > prefetch_data_;
  shared_ptr<Blob<Dtype> > prefetch_label_;
  Blob<Dtype> data_mean_;
  vector<std::pair<std::string, vector<int> > > image_database_;
  enum WindowField { IMAGE_INDEX, LABEL, OVERLAP, X1, Y1, X2, Y2, NUM };
  vector<vector<float> > fg_windows_;
  vector<vector<float> >::iterator iter_;
  int item_id;
 // -------------------- no back ground windows ------------------------
//  vector<vector<float> > bg_windows_;
};

void FillInOffsets(int *w, int *h, int width, int height, int crop_size);

}  // namespace caffe

#endif  // CAFFE_DATA_LAYERS_HPP_
