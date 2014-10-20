// Copyright 2014 BVLC and contributors.

#include <stdint.h>
#include <leveldb/db.h>
#include "leveldb/cache.h"
#include "leveldb/options.h"
#include <pthread.h>

#include <string>
#include <vector>
#include <fstream>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/vision_layers.hpp"

using std::string;

namespace caffe {

template <typename Dtype>
void* DataPool5LayerPrefetch(void* layer_pointer) {
  CHECK(layer_pointer);
  DataPool5Layer<Dtype>* layer = static_cast<DataPool5Layer<Dtype>*>(layer_pointer);
  CHECK(layer);
  Pool5Feature datum;
  CHECK(layer->prefetch_data_);
  Dtype* top_data = layer->prefetch_data_->mutable_cpu_data();
  Dtype* top_label;
  if (layer->output_labels_) {
    top_label = layer->prefetch_label_->mutable_cpu_data();
  }
  const int batch_size = layer->layer_param_.data_pool5_param().batch_size();
  const bool shuffle = layer->layer_param_.data_pool5_param().shuffle();
  const float fg_fraction =
      layer->layer_param_.data_pool5_param().fg_fraction();
  const float fg_threshold =
        layer->layer_param_.data_pool5_param().fg_threshold();
  const float bg_threshold =
        layer->layer_param_.data_pool5_param().bg_threshold();

  // datum scales
  const int channels = layer->datum_channels_;
  const int height = layer->datum_height_;
  const int width = layer->datum_width_;
  const int size = layer->datum_size_;

  const int num_fg = static_cast<int>(static_cast<float>(batch_size) * fg_fraction);
  // NOTE!!------------------- here shuffle is used for randomization -----------------
  // if (!shuffle) {
  //   for (int item_id = 0; item_id < batch_size; ++item_id) {
  //     // get a blob
  //     CHECK(layer->iter_);
  //     CHECK(layer->iter_->Valid());
  //     datum.ParseFromString(layer->iter_->value().ToString());
  //     int ft_dim = datum.dim_feature();
  //     for (int j = 0; j < ft_dim; j++) {
  //         top_data[item_id * size + j] = datum.feat(j);
  //         // if (j < 10)
  //         //   std::cout << datum.feat(j) << " ";
  //     }

  //     if (layer->output_labels_) {
  //       top_label[item_id] = (int)datum.label();
  //       //std::cout << datum.label() << std::endl;
  //     }
  //     // go to the next iter
  //     layer->iter_->Next();
  //     if (!layer->iter_->Valid()) {
  //       // We have reached the end. Restart from the first.
  //       DLOG(INFO) << "Restarting data prefetching from start.";
  //       layer->iter_->SeekToFirst();
  //     }
  //   }
  // }
  {
    int item_id = 0;
    int fg_size = layer->fg_keys_.size();
    int bg_size = layer->bg_keys_.size();
    // fill in fg samples
    int i = 0;
    while (i < num_fg) {
      // valid fg sample
      if (layer->fg_overlaps_[layer->fg_counter_] >= fg_threshold) {
        CHECK(layer->fg_iter_);
        CHECK(layer->fg_iter_->Valid());
        datum.ParseFromString(layer->fg_iter_->value().ToString());
        // string k = layer->fg_keys_[layer->fg_counter_++];
        // string value;
        // layer->fgdb_->Get(leveldb::ReadOptions(), k, &value);
        // datum.ParseFromString(value);
        int ft_dim = datum.dim_feature();
        for (int j = 0; j < ft_dim; j++) {
            top_data[item_id * size + j] = datum.feat(j);
        }
        if (layer->output_labels_) {
          top_label[item_id] = (int)datum.label();
        }
        item_id++;
        i++;
      }
      layer->fg_counter_++;
      layer->fg_iter_->Next();
      layer->fg_counter_ %= fg_size;
      if (!layer->fg_iter_->Valid()) {
        // We have reached the end. Restart from the first.
        DLOG(INFO) << "Restarting fg data prefetching from start.";
        layer->fg_iter_->SeekToFirst();
      }
    }
    //LOG(INFO) << "fg samples loaded" << std::endl;
    // fill in bg samples
    i = 0;
    while (i < batch_size - num_fg) {
      if (layer->bg_overlaps_[layer->bg_counter_] <= bg_threshold) {
        CHECK(layer->bg_iter_);
        CHECK(layer->bg_iter_->Valid());
        datum.ParseFromString(layer->bg_iter_->value().ToString());
        int ft_dim = datum.dim_feature();
        for (int j = 0; j < ft_dim; j++) {
            top_data[item_id * size + j] = datum.feat(j);
        }
        if (layer->output_labels_) {
          top_label[item_id] = (int)datum.label();
        }
        item_id++;
        i++;
      }
      layer->bg_counter_++;
      layer->bg_iter_->Next();
      layer->bg_counter_ %= bg_size;
      if (!layer->bg_iter_->Valid()) {
        DLOG(INFO) << "Restarting bg data prefetching from start.";
        layer->bg_iter_->SeekToFirst();
      }
    }
  }

  return static_cast<void*>(NULL);
}

template <typename Dtype>
void DataPool5Layer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 0) << "Data Layer takes no input blobs.";
  CHECK_GE(top->size(), 1) << "Data Layer takes at least one blob as output.";
  CHECK_LE(top->size(), 2) << "Data Layer takes at most two blobs as output.";
  if (top->size() == 1) {
    this->output_labels_ = false;
  } else {
    this->output_labels_ = true;
  }

  LOG(INFO) << "Pool5 Data Layer:" << std::endl
      << "  foreground (object) overlap threshold: "
      << this->layer_param_.data_pool5_param().fg_threshold() << std::endl
      << "  background (non-object) overlap threshold: "
      << this->layer_param_.data_pool5_param().bg_threshold() << std::endl
      << "  foreground sampling fraction: "
      << this->layer_param_.data_pool5_param().fg_fraction();

  const float fg_threshold =
        this->layer_param_.data_pool5_param().fg_threshold();
  const float bg_threshold =
        this->layer_param_.data_pool5_param().bg_threshold();
  // Initialize the leveldb
  leveldb::DB* db_temp;
  leveldb::Options options;
  options.block_cache = leveldb::NewLRUCache(1000 * 1048576);
  options.create_if_missing = false;
  options.max_open_files = 100;
  string fgdb(this->layer_param_.data_pool5_param().source() + "_fg_leveldb");
  string bgdb(this->layer_param_.data_pool5_param().source() + "_bg_leveldb");

  LOG(INFO) << "Opening leveldb " << fgdb;
  leveldb::Status status = leveldb::DB::Open(options, fgdb, &db_temp);
  CHECK(status.ok()) << "Failed to open leveldb " << fgdb  << std::endl
      << status.ToString();
  this->fg_db_.reset(db_temp);
  this->fg_iter_.reset(this->fg_db_->NewIterator(leveldb::ReadOptions()));
  this->fg_iter_->SeekToFirst();

  LOG(INFO) << "Opening leveldb " << bgdb;
  status = leveldb::DB::Open(options, bgdb, &db_temp);
  CHECK(status.ok()) << "Failed to open leveldb " << bgdb  << std::endl
      << status.ToString();
  this->bg_db_.reset(db_temp);
  this->bg_iter_.reset(this->bg_db_->NewIterator(leveldb::ReadOptions()));
  this->bg_iter_->SeekToFirst();

  // Load keys for leveldb  and overlap
  string fgkeyfile(this->layer_param_.data_pool5_param().source() + "_fg_keys.txt");
  string fgoverlapfile(this->layer_param_.data_pool5_param().source() + "_fg_overlaps.txt");
  LOG(INFO) << "Loading keys for leveldb " << fgkeyfile << std::endl;
  LOG(INFO) << "Loading overlaps for leveldb " << fgoverlapfile << std::endl;
  std::ifstream fginfile(fgkeyfile.c_str());
  std::ifstream fginfile2(fgoverlapfile.c_str());
  CHECK(fginfile.good()) << "Failed to open key file " << fgkeyfile << std::endl;
  CHECK(fginfile2.good()) << "Failed to open overlap file " << fgoverlapfile << std::endl;
  string key;
  float ov;
  while (fginfile >> key) {
    fginfile2 >> ov;
    //if (ov >= fg_threshold) {
    fg_keys_.push_back(key);
    fg_overlaps_.push_back(ov);
    //}
  }

  string bgkeyfile(this->layer_param_.data_pool5_param().source() + "_bg_keys.txt");
  string bgoverlapfile(this->layer_param_.data_pool5_param().source() + "_bg_overlaps.txt");
  LOG(INFO) << "Loading keys for leveldb " << bgkeyfile << std::endl;
  LOG(INFO) << "Loading overlaps for leveldb " << bgoverlapfile << std::endl;
  std::ifstream bginfile(bgkeyfile.c_str());
  std::ifstream bginfile2(bgoverlapfile.c_str());
  CHECK(bginfile.good()) << "Failed to open key file " << bgkeyfile << std::endl;
  CHECK(bginfile2.good()) << "Failed to open overlap file " << bgoverlapfile << std::endl;
  while (bginfile >> key) {
    bginfile2 >> ov;
    //if (ov < bg_threshold) {
    bg_keys_.push_back(key);
    bg_overlaps_.push_back(ov);
    //}
  }


  LOG(INFO) << "fg keys: " << fg_keys_.size() << std::endl
    << "bg keys: " << bg_keys_.size() << std::endl;
  fginfile.close(); bginfile.close();
  fginfile2.close(); bginfile2.close();
  fg_counter_ = 0;
  bg_counter_ = 0;
  // std::random_shuffle(fg_keys_.begin(), fg_keys_.end());
  // std::random_shuffle(bg_keys_.begin(), bg_keys_.end());
  LOG(INFO) << "keys shuffled" << std::endl;
  // Read a data point, and use it to initialize the top blob.
  Pool5Feature datum;
  datum.ParseFromString(this->fg_iter_->value().ToString());
  // pool5 feature dim
  (*top)[0]->Reshape(
      this->layer_param_.data_pool5_param().batch_size(), datum.dim_feature(),
      1, 1);
  this->prefetch_data_.reset(new Blob<Dtype>(
      this->layer_param_.data_pool5_param().batch_size(), datum.dim_feature(),
      1, 1));

  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();
  // label
  if (this->output_labels_) {
    (*top)[1]->Reshape(this->layer_param_.data_pool5_param().batch_size(), 1, 1, 1);
    this->prefetch_label_.reset(
        new Blob<Dtype>(this->layer_param_.data_pool5_param().batch_size(), 1, 1, 1));
  }
  // datum size
  this->datum_channels_ = datum.dim_feature();
  this->datum_height_ = 1;
  this->datum_width_ = 1;
  this->datum_size_ = datum.dim_feature();

  // Now, start the prefetch thread. Before calling prefetch, we make two
  // cpu_data calls so that the prefetch thread does not accidentally make
  // simultaneous cudaMalloc calls when the main thread is running. In some
  // GPUs this seems to cause failures if we do not so.
  this->prefetch_data_->mutable_cpu_data();
  if (this->output_labels_) {
    this->prefetch_label_->mutable_cpu_data();
  }
  DLOG(INFO) << "Initializing prefetch";
  CreatePrefetchThread();
  DLOG(INFO) << "Prefetch initialized.";
}

template <typename Dtype>
void DataPool5Layer<Dtype>::CreatePrefetchThread() {
  this->phase_ = Caffe::phase();
  const bool prefetch_needs_rand = false;
  if (prefetch_needs_rand) {
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    this->prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
  } else {
    this->prefetch_rng_.reset();
  }
  // Create the thread.
  CHECK(!pthread_create(&(this->thread_), NULL, DataPool5LayerPrefetch<Dtype>,
        static_cast<void*>(this))) << "Pthread execution failed.";
}

INSTANTIATE_CLASS(DataPool5Layer);

}  // namespace caffe
