// Copyright 2014 BVLC and contributors.

#include <stdint.h>
#include <leveldb/db.h>
#include <pthread.h>

#include <string>
#include <vector>
#include <cstdio>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/vision_layers.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;
using std::string;

namespace caffe {

template <typename Dtype>
void* CompactDataLayerPrefetch(void* layer_pointer) {
  CHECK(layer_pointer);
  CompactDataLayer<Dtype>* layer = static_cast<CompactDataLayer<Dtype>*>(layer_pointer);
  CHECK(layer);
  CHECK(layer->prefetch_data_);
  Dtype* top_data = layer->prefetch_data_->mutable_cpu_data();
  Dtype* top_label;
  if (layer->output_labels_) {
    top_label = layer->prefetch_label_->mutable_cpu_data();
  }
  const Dtype scale = layer->layer_param_.data_param().scale();
  const int batch_size = layer->layer_param_.data_param().batch_size();
  const int crop_size = layer->layer_param_.data_param().crop_size();
  const bool mirror = layer->layer_param_.data_param().mirror();

  CHECK_GT(crop_size, 0) << "crop size must be greater than 0";
  // datum scales
  const Dtype* mean = layer->data_mean_.cpu_data();
  int w[5], h[5];

  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a image
    CHECK(layer->iter_);
    CHECK(layer->iter_->Valid());
    string value = layer->iter_->value().ToString();
    CvMat mat = cvMat(1, 1000 * 1000, CV_8UC1, const_cast<char *>(value.data()) + sizeof(int));
    // if (item_id == batch_size -1)
    //   LOG(INFO) << layer->iter_->key().ToString() << " " << *((int *)const_cast<char *>(value.data()));
    IplImage *img = cvDecodeImage(&mat, 1);
    int channels = img->nChannels;
    int width = img->width;
    int height = img->height;
    unsigned char* data = (unsigned char *)img->imageData;
    int step = img->widthStep / sizeof(char);
    // crop 4 corners + center
    caffe::FillInOffsets(w, h, width, height, crop_size);
    //LOG(INFO) << "height: " << height << ", width: " << width;
    int h_off, w_off;
    // We only do random crop when we do training.
    if (mirror) {
      int r = layer->PrefetchRand() % 5;
      h_off = h[r];
      w_off = w[r];
    } else {
      h_off = h[4];
      w_off = w[4];
    }
    if (mirror && layer->PrefetchRand() % 2) {
      // Copy mirrored version
      //LOG(INFO) << "mirror " << h_off << ", " << w_off;
      for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < crop_size; ++h) {
          for (int w = 0; w < crop_size; ++w) {
            int top_index = ((item_id * channels + c) * crop_size + h)
                            * crop_size + (crop_size - 1 - w);
            int data_index = (h + h_off) * step + (w + w_off) * channels + c;
            int mean_index = (c * crop_size + h) * crop_size + w;
            Dtype datum_element = static_cast<Dtype>(data[data_index]);
            top_data[top_index] = (datum_element - mean[mean_index]) * scale;
            // if (h < 5 && w < 5)
            //   cout << datum_element << " ";
          }
          // if (h < 5) cout << endl;
        }
      }
    } else {
      // Normal copy
      //LOG(INFO) << "not mirror " << h_off << ", " << w_off;
      for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < crop_size; ++h) {
          for (int w = 0; w < crop_size; ++w) {
            int top_index = ((item_id * channels + c) * crop_size + h)
                            * crop_size + w;
            int data_index = (h + h_off) * step + (w + w_off) * channels + c;
            int mean_index = (c * crop_size + h) * crop_size + w;
            Dtype datum_element = static_cast<Dtype>(data[data_index]);
            top_data[top_index] = (datum_element - mean[mean_index]) * scale;
          // if (h < 5 && w < 5)
          //     cout << datum_element << " ";
          }
          // if (h < 5) cout << endl;
        }
      }
    }
    cvReleaseImage(&img);  // release current image

    if (layer->output_labels_) {
      //top_label[item_id] = datum.label();
      top_label[item_id] = *((int *)const_cast<char *>(value.data()));
      // LOG(INFO) << "label: " << top_label[item_id];
    }
    //getchar();
    // go to the next iter
    layer->iter_->Next();
    if (!layer->iter_->Valid()) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      layer->iter_->SeekToFirst();
    }
  }

  return static_cast<void*>(NULL);
}


template <typename Dtype>
void CompactDataLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 0) << "Data Layer takes no input blobs.";
  CHECK_GE(top->size(), 1) << "Data Layer takes at least one blob as output.";
  CHECK_LE(top->size(), 2) << "Data Layer takes at most two blobs as output.";
  if (top->size() == 1) {
    this->output_labels_ = false;
  } else {
    this->output_labels_ = true;
  }
  // Initialize the leveldb
  leveldb::DB* db_temp;
  leveldb::Options options;
  options.create_if_missing = false;
  options.max_open_files = 100;
  LOG(INFO) << "Opening leveldb " << this->layer_param_.data_param().source();
  leveldb::Status status = leveldb::DB::Open(
      options, this->layer_param_.data_param().source(), &db_temp);
  CHECK(status.ok()) << "Failed to open leveldb "
      << this->layer_param_.data_param().source() << std::endl
      << status.ToString();
  this->db_.reset(db_temp);
  this->iter_.reset(this->db_->NewIterator(leveldb::ReadOptions()));
  this->iter_->SeekToFirst();
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
                        this->layer_param_.data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    while (skip-- > 0) {
      this->iter_->Next();
      if (!this->iter_->Valid()) {
        this->iter_->SeekToFirst();
      }
    }
  }
  // Read a data point, and use it to initialize the top blob.
  // TODO: read an image from the DB
  string value = this->iter_->value().ToString();
  //std::cout << this->iter_->key().ToString() << std::endl;
  CvMat mat = cvMat(1, 1000 * 1000, CV_8UC1, const_cast<char *>(value.data()) + sizeof(int));
  IplImage *img = cvDecodeImage(&mat, 1);
  this->datum_channels_ = img->nChannels;
  this->datum_height_ = img->height;
  this->datum_width_ = img->width;
  cvReleaseImage(&img);

  int crop_size = this->layer_param_.data_param().crop_size();
  CHECK_GT(crop_size, 0) << "crop size must be greater than 0";
  (*top)[0]->Reshape(this->layer_param_.data_param().batch_size(),
                       this->datum_channels_, crop_size, crop_size);
  this->prefetch_data_.reset(new Blob<Dtype>(
        this->layer_param_.data_param().batch_size(), this->datum_channels_,
        crop_size, crop_size));

  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();
  // label
  if (this->output_labels_) {
    (*top)[1]->Reshape(this->layer_param_.data_param().batch_size(), 1, 1, 1);
    this->prefetch_label_.reset(
        new Blob<Dtype>(this->layer_param_.data_param().batch_size(), 1, 1, 1));
  }
  // sanity check
  CHECK_GE(this->datum_height_, crop_size);
  CHECK_GE(this->datum_width_, crop_size);
  // check if we want to have mean
  this->data_mean_.Reshape(1, this->datum_channels_, crop_size, crop_size);
  CHECK(this->layer_param_.data_param().has_mean_file());
  const string& mean_file = this->layer_param_.data_param().mean_file();
  LOG(INFO) << "Loading mean file from" << mean_file;
  BlobProto blob_proto;
  ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
  Blob<Dtype> tmp;
  tmp.FromProto(blob_proto);
  const Dtype* src_data = tmp.cpu_data();
  Dtype* dst_data = this->data_mean_.mutable_cpu_data();
  CHECK_EQ(tmp.num(), 1);
  CHECK_EQ(tmp.channels(), this->datum_channels_);
  CHECK_GE(tmp.height(), crop_size);
  CHECK_GE(tmp.width(), crop_size);
  int w_off = (tmp.width() - crop_size) / 2;
  int h_off = (tmp.height() - crop_size) / 2;
  for (int c = 0; c < this->datum_channels_; c++) {
    for (int h = 0; h < crop_size; h++) {
      for (int w = 0; w < crop_size; w++) {
        int src_idx = (c * tmp.height() + h + h_off) * tmp.width() + w + w_off;
        int dst_idx = (c * crop_size + h) * crop_size + w;
        dst_data[dst_idx] = src_data[src_idx];
      }
    }
  }
  // Now, start the prefetch thread. Before calling prefetch, we make two
  // cpu_data calls so that the prefetch thread does not accidentally make
  // simultaneous cudaMalloc calls when the main thread is running. In some
  // GPUs this seems to cause failures if we do not so.
  this->prefetch_data_->mutable_cpu_data();
  if (this->output_labels_) {
    this->prefetch_label_->mutable_cpu_data();
  }
  this->data_mean_.cpu_data();
  DLOG(INFO) << "Initializing prefetch";
  this->CreatePrefetchThread();
  DLOG(INFO) << "Prefetch initialized.";
}


template <typename Dtype>
void CompactDataLayer<Dtype>::CreatePrefetchThread() {
  this->phase_ = Caffe::phase();
  const bool prefetch_needs_rand = (this->phase_ == Caffe::TRAIN) &&
      (this->layer_param_.data_param().mirror() ||
       this->layer_param_.data_param().crop_size());
  if (prefetch_needs_rand) {
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    this->prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
  } else {
    this->prefetch_rng_.reset();
  }
  // Create the thread.
  CHECK(!pthread_create(&(this->thread_), NULL, CompactDataLayerPrefetch<Dtype>,
        static_cast<void*>(this))) << "Pthread execution failed.";
}

void FillInOffsets(int *w, int *h, int width, int height, int crop_size) {
  w[0] = 0; h[0] = 0;
  w[1] = 0; h[1] = height - crop_size;
  w[2] = width - crop_size; h[2] = 0;
  w[3] = width - crop_size; h[3] = height - crop_size;
  w[4] = (width - crop_size) / 2; h[4] = (height - crop_size) / 2;
}


INSTANTIATE_CLASS(CompactDataLayer);

}  // namespace caffe
