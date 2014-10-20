// Copyright 2014 BVLC and contributors.
//
// Based on data_layer.cpp by Yangqing Jia.

#include <stdint.h>
#include <pthread.h>

#include <algorithm>
#include <string>
#include <vector>
#include <map>
#include <fstream>  // NOLINT(readability/streams)
#include <utility>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/vision_layers.hpp"

using std::string;
using std::map;
using std::pair;

// caffe.proto > LayerParameter > WindowDataParameter
//   'source' field specifies the window_file
//   'crop_size' indicates the desired warped size

namespace caffe {

template <typename Dtype>
void* WindowDataNoBgLayerPrefetch(void* layer_pointer) {
  WindowDataNoBgLayer<Dtype>* layer =
      reinterpret_cast<WindowDataNoBgLayer<Dtype>*>(layer_pointer);

  // At each iteration, sample N windows where N*p are foreground (object)
  // windows and N*(1-p) are background (non-object) windows

  Dtype* top_data = layer->prefetch_data_->mutable_cpu_data();
  Dtype* top_label = layer->prefetch_label_->mutable_cpu_data();
  const Dtype scale = layer->layer_param_.window_data_param().scale();
  const int batch_size = layer->layer_param_.window_data_param().batch_size();
  const int crop_size = layer->layer_param_.window_data_param().crop_size();
  const int context_pad = layer->layer_param_.window_data_param().context_pad();
  const bool mirror = layer->layer_param_.window_data_param().mirror();
  const float fg_fraction =
      layer->layer_param_.window_data_param().fg_fraction();
  const Dtype* mean = layer->data_mean_.cpu_data();
  const int mean_off = (layer->data_mean_.width() - crop_size) / 2;
  const int mean_width = layer->data_mean_.width();
  const int mean_height = layer->data_mean_.height();
  cv::Size cv_crop_size(crop_size, crop_size);
  const string& crop_mode = layer->layer_param_.window_data_param().crop_mode();

  bool use_square = (crop_mode == "square") ? true : false;

  // zero out batch
  memset(top_data, 0, sizeof(Dtype)*layer->prefetch_data_->count());

  const int num_fg = static_cast<int>(static_cast<float>(batch_size)
      * fg_fraction);
  const int num_samples[2] = { batch_size - num_fg, num_fg };

  int item_id = 0;
  // sample from bg set then fg set
  for (int is_fg = 0; is_fg < 2; ++is_fg) {
    for (int dummy = 0; dummy < num_samples[is_fg]; ++dummy) {
      // sample a window
      const unsigned int rand_index = layer->PrefetchRand();
      vector<float> window = (is_fg) ?
          layer->fg_windows_[rand_index % layer->fg_windows_.size()] :
          layer->bg_windows_[rand_index % layer->bg_windows_.size()];

      bool do_mirror = false;
      if (mirror && layer->PrefetchRand() % 2) {
        do_mirror = true;
      }

      // load the image containing the window
      pair<std::string, vector<int> > image =
          layer->image_database_[window[WindowDataLayer<Dtype>::IMAGE_INDEX]];

      cv::Mat cv_img = cv::imread(image.first, CV_LOAD_IMAGE_COLOR);
      if (!cv_img.data) {
        LOG(ERROR) << "Could not open or find file " << image.first;
        return reinterpret_cast<void*>(NULL);
      }
      const int channels = cv_img.channels();

      // crop window out of image and warp it
      int x1 = window[WindowDataLayer<Dtype>::X1];
      int y1 = window[WindowDataLayer<Dtype>::Y1];
      int x2 = window[WindowDataLayer<Dtype>::X2];
      int y2 = window[WindowDataLayer<Dtype>::Y2];

      int pad_w = 0;
      int pad_h = 0;
      if (context_pad > 0 || use_square) {
        // scale factor by which to expand the original region
        // such that after warping the expanded region to crop_size x crop_size
        // there's exactly context_pad amount of padding on each side
        Dtype context_scale = static_cast<Dtype>(crop_size) /
            static_cast<Dtype>(crop_size - 2*context_pad);

        // compute the expanded region
        Dtype half_height = static_cast<Dtype>(y2-y1+1)/2.0;
        Dtype half_width = static_cast<Dtype>(x2-x1+1)/2.0;
        Dtype center_x = static_cast<Dtype>(x1) + half_width;
        Dtype center_y = static_cast<Dtype>(y1) + half_height;
        if (use_square) {
          if (half_height > half_width) {
            half_width = half_height;
          } else {
            half_height = half_width;
          }
        }
        x1 = static_cast<int>(round(center_x - half_width*context_scale));
        x2 = static_cast<int>(round(center_x + half_width*context_scale));
        y1 = static_cast<int>(round(center_y - half_height*context_scale));
        y2 = static_cast<int>(round(center_y + half_height*context_scale));

        // the expanded region may go outside of the image
        // so we compute the clipped (expanded) region and keep track of
        // the extent beyond the image
        int unclipped_height = y2-y1+1;
        int unclipped_width = x2-x1+1;
        int pad_x1 = std::max(0, -x1);
        int pad_y1 = std::max(0, -y1);
        int pad_x2 = std::max(0, x2 - cv_img.cols + 1);
        int pad_y2 = std::max(0, y2 - cv_img.rows + 1);
        // clip bounds
        x1 = x1 + pad_x1;
        x2 = x2 - pad_x2;
        y1 = y1 + pad_y1;
        y2 = y2 - pad_y2;
        CHECK_GT(x1, -1);
        CHECK_GT(y1, -1);
        CHECK_LT(x2, cv_img.cols);
        CHECK_LT(y2, cv_img.rows);

        int clipped_height = y2-y1+1;
        int clipped_width = x2-x1+1;

        // scale factors that would be used to warp the unclipped
        // expanded region
        Dtype scale_x =
            static_cast<Dtype>(crop_size)/static_cast<Dtype>(unclipped_width);
        Dtype scale_y =
            static_cast<Dtype>(crop_size)/static_cast<Dtype>(unclipped_height);

        // size to warp the clipped expanded region to
        cv_crop_size.width =
            static_cast<int>(round(static_cast<Dtype>(clipped_width)*scale_x));
        cv_crop_size.height =
            static_cast<int>(round(static_cast<Dtype>(clipped_height)*scale_y));
        pad_x1 = static_cast<int>(round(static_cast<Dtype>(pad_x1)*scale_x));
        pad_x2 = static_cast<int>(round(static_cast<Dtype>(pad_x2)*scale_x));
        pad_y1 = static_cast<int>(round(static_cast<Dtype>(pad_y1)*scale_y));
        pad_y2 = static_cast<int>(round(static_cast<Dtype>(pad_y2)*scale_y));

        pad_h = pad_y1;
        // if we're mirroring, we mirror the padding too (to be pedantic)
        if (do_mirror) {
          pad_w = pad_x2;
        } else {
          pad_w = pad_x1;
        }

        // ensure that the warped, clipped region plus the padding fits in the
        // crop_size x crop_size image (it might not due to rounding)
        if (pad_h + cv_crop_size.height > crop_size) {
          cv_crop_size.height = crop_size - pad_h;
        }
        if (pad_w + cv_crop_size.width > crop_size) {
          cv_crop_size.width = crop_size - pad_w;
        }
      }

      cv::Rect roi(x1, y1, x2-x1+1, y2-y1+1);
      cv::Mat cv_cropped_img = cv_img(roi);
      cv::resize(cv_cropped_img, cv_cropped_img,
          cv_crop_size, 0, 0, cv::INTER_LINEAR);

      // horizontal flip at random
      if (do_mirror) {
        cv::flip(cv_cropped_img, cv_cropped_img, 1);
      }

      // copy the warped window into top_data
      for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < cv_cropped_img.rows; ++h) {
          for (int w = 0; w < cv_cropped_img.cols; ++w) {
            Dtype pixel =
                static_cast<Dtype>(cv_cropped_img.at<cv::Vec3b>(h, w)[c]);

            top_data[((item_id * channels + c) * crop_size + h + pad_h)
                     * crop_size + w + pad_w]
                = (pixel
                    - mean[(c * mean_height + h + mean_off + pad_h)
                           * mean_width + w + mean_off + pad_w])
                  * scale;
          }
        }
      }

      // get window label
      // !! -------------------------------------------------------------------------
      top_label[item_id] = window[WindowDataLayer<Dtype>::LABEL] - 1;
      // !! --------------- minus one since we do not want bg windows ------------- !!

      #if 0
      // useful debugging code for dumping transformed windows to disk
      string file_id;
      std::stringstream ss;
      ss << layer->PrefetchRand();
      ss >> file_id;
      std::ofstream inf((string("dump/") + file_id +
          string("_info.txt")).c_str(), std::ofstream::out);
      inf << image.first << std::endl
          << window[WindowDataLayer<Dtype>::X1]+1 << std::endl
          << window[WindowDataLayer<Dtype>::Y1]+1 << std::endl
          << window[WindowDataLayer<Dtype>::X2]+1 << std::endl
          << window[WindowDataLayer<Dtype>::Y2]+1 << std::endl
          << do_mirror << std::endl
          << top_label[item_id] << std::endl
          << is_fg << std::endl;
      inf.close();
      std::ofstream top_data_file((string("dump/") + file_id +
          string("_data.txt")).c_str(),
          std::ofstream::out | std::ofstream::binary);
      for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < crop_size; ++h) {
          for (int w = 0; w < crop_size; ++w) {
            top_data_file.write(reinterpret_cast<char*>(
                &top_data[((item_id * channels + c) * crop_size + h)
                          * crop_size + w]),
                sizeof(Dtype));
          }
        }
      }
      top_data_file.close();
      #endif

      item_id++;
    }
  }

  return reinterpret_cast<void*>(NULL);
}


template <typename Dtype>
void WindowDataNoBgLayer<Dtype>::CreatePrefetchThread() {
  const bool prefetch_needs_rand =
      this->layer_param_.window_data_param().mirror() ||
      this->layer_param_.window_data_param().crop_size();
  if (prefetch_needs_rand) {
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    this->prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
  } else {
    this->prefetch_rng_.reset();
  }
  // Create the thread.
  CHECK(!pthread_create(&(this->thread_), NULL, WindowDataNoBgLayerPrefetch<Dtype>,
        static_cast<void*>(this))) << "Pthread execution failed.";
}

INSTANTIATE_CLASS(WindowDataNoBgLayer);

}  // namespace caffe
