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
void* PrefetchHelper(void* arg) {
    void** args = (void**)arg;
    WindowDataFeatExtractLayer<Dtype>* layer =
        reinterpret_cast<WindowDataFeatExtractLayer<Dtype>*>(args[0]);
    vector<float> window;
    int item_id = *reinterpret_cast<int*>(args[1]);
    // std::cout << item_id << std::endl;
    if (item_id % 2 == 0)
      window = *(layer->iter_);
    else
      if (layer->iter_ + 1 != layer->fg_windows_.end())
        window = *(layer->iter_ + 1);
      else
        return reinterpret_cast<void*>(NULL);

    Dtype* top_data = layer->prefetch_data_->mutable_cpu_data();
    const Dtype scale = layer->layer_param_.window_data_param().scale();
    const int batch_size = layer->layer_param_.window_data_param().batch_size();
    const int crop_size = layer->layer_param_.window_data_param().crop_size();
    const int context_pad = layer->layer_param_.window_data_param().context_pad();
    const Dtype* mean = layer->data_mean_.cpu_data();
    const int mean_off = (layer->data_mean_.width() - crop_size) / 2;
    const int mean_width = layer->data_mean_.width();
    const int mean_height = layer->data_mean_.height();

    cv::Size cv_crop_size(crop_size, crop_size);
    pair<std::string, vector<int> > image =
      layer->image_database_[window[WindowDataFeatExtractLayer<Dtype>::IMAGE_INDEX]];
    cv::Mat cv_img = cv::imread(image.first, CV_LOAD_IMAGE_COLOR);
    if (!cv_img.data) {
      LOG(ERROR) << "Cound not open or find file " << image.first;
      return reinterpret_cast<void*>(NULL);
    }
    const int channels = cv_img.channels();
    // crop window out of image and warp it
    int x1 = window[WindowDataFeatExtractLayer<Dtype>::X1];
    int y1 = window[WindowDataFeatExtractLayer<Dtype>::Y1];
    int x2 = window[WindowDataFeatExtractLayer<Dtype>::X2];
    int y2 = window[WindowDataFeatExtractLayer<Dtype>::Y2];
    x1 = std::max(0, x1);
    x1 = std::min(cv_img.cols - 1, x1);
    x2 = std::max(0, x2);
    x2 = std::min(cv_img.cols - 1, x2);
    x1 = std::min(x1, x2); x2 = std::max(x1, x2);
    y1 = std::max(0, y1);
    y1= std::min(cv_img.rows - 1, y1);
    y2 = std::max(0, y2);
    y2 = std::min(cv_img.rows - 1, y2);
    y1 = std::min(y1, y2); y2 = std::max(y1, y2);
    // std::cout << "x1, x2:" << x1 << ", " << x2 <<  std::endl;
    // std::cout << "y1, y2:" << y1 << ", " << y2 <<  std::endl;


    int pad_w = 0;
    int pad_h = 0;

    Dtype context_scale = static_cast<Dtype>(crop_size) /
        static_cast<Dtype>(crop_size - 2*context_pad);
    // compute the expanded region
    Dtype half_height = static_cast<Dtype>(y2-y1+1)/2.0;
    Dtype half_width = static_cast<Dtype>(x2-x1+1)/2.0;
    Dtype center_x = static_cast<Dtype>(x1) + half_width;
    Dtype center_y = static_cast<Dtype>(y1) + half_height;

    x1 = static_cast<int>(round(center_x - half_width*context_scale));
    x2 = static_cast<int>(round(center_x + half_width*context_scale));
    y1 = static_cast<int>(round(center_y - half_height*context_scale));
    y2 = static_cast<int>(round(center_y + half_height*context_scale));

    // the expanded region may go outside of the image
    // so we compute the clipped (expanded) region and keep track of
    // the extent beyond the image
    // std::cout << "x1, x2:" << x1 << ", " << x2 <<  std::endl;
    // std::cout << center_y << ", " << half_height << std::endl;
    // std::cout << "y1, y2:" << y1 << ", " << y2 <<  std::endl;
    int unclipped_height = y2-y1+1;
    int unclipped_width = x2-x1+1;
    int pad_x1 = std::max(0, -x1);
    int pad_y1 = std::max(0, -y1);
    // clip bounds
    x1 = std::max(0, x1);
    x2 = std::min(cv_img.cols - 1, x2);
    y1 = std::max(0, y1);
    y2 = std::min(cv_img.rows - 1, y2);
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
        // std::cout << x1 << "::" << x2 << std::endl;
        // std::cout << y1 << "::" << y2<< std::endl;
        // std::cout << cv_img.cols << ",," << cv_img.rows << std::endl;
    // size to warp the clipped expanded region to
    cv_crop_size.width =
        static_cast<int>(round(static_cast<Dtype>(clipped_width)*scale_x));
    cv_crop_size.height =
        static_cast<int>(round(static_cast<Dtype>(clipped_height)*scale_y));
    pad_x1 = static_cast<int>(round(static_cast<Dtype>(pad_x1)*scale_x));
    pad_y1 = static_cast<int>(round(static_cast<Dtype>(pad_y1)*scale_y));
    // std::cout << "crop_w, crop_h:" << cv_crop_size.width << ", " << cv_crop_size.height << std::endl;
    // std::cout << "pad_x1, scale_x:" << pad_x1 << "," << scale_x << std::endl;
    // std::cout << "pad_y1, scale_y:" << pad_y1 << "," << scale_y << std::endl;
    pad_h = pad_y1;
    pad_w = pad_x1;

    // ensure that the warped, clipped region plus the padding fits in the
    // crop_size x crop_size image (it might not due to rounding)
    if (pad_h + cv_crop_size.height > crop_size) {
      cv_crop_size.height = crop_size - pad_h;
    }
    if (pad_w + cv_crop_size.width > crop_size) {
      cv_crop_size.width = crop_size - pad_w;
    }

    cv::Rect roi(x1, y1, x2-x1+1, y2-y1+1);
    cv::Mat cv_cropped_img = cv_img(roi);

    cv::resize(cv_cropped_img, cv_cropped_img,
      cv_crop_size, 0, 0, cv::INTER_LINEAR);

    // copy the warped window into top_data
    for (int c = 0; c < channels; ++c) {
      for (int h = 0; h < cv_cropped_img.rows; ++h) {
        for (int w = 0; w < cv_cropped_img.cols; ++w) {
          Dtype pixel =
              static_cast<Dtype>(cv_cropped_img.at<cv::Vec3b>(h, w)[c]);

          top_data[((item_id * channels + c) * crop_size + h + pad_h)
                   * crop_size + w + pad_w]
              // = pixel;
              = (pixel
                  - mean[(c * mean_height + h + mean_off + pad_h)
                         * mean_width + w + mean_off + pad_w])
                * scale;
        }
      }
    }
    return reinterpret_cast<void*>(NULL);
}

template <typename Dtype>
void* WindowDataFeatExtractLayerPrefetch2(void* layer_pointer) {
    WindowDataFeatExtractLayer<Dtype>* layer =
        reinterpret_cast<WindowDataFeatExtractLayer<Dtype>*>(layer_pointer);

    const int batch_size = layer->layer_param_.window_data_param().batch_size();
    Dtype* top_data = layer->prefetch_data_->mutable_cpu_data();
    memset(top_data, 0, sizeof(Dtype)*layer->prefetch_data_->count());
    const int num_fg = static_cast<int>(batch_size);

    std::cout << "window num: " << layer->fg_windows_.size() << std::endl;
    std::cout << "num_fg: " << num_fg << std::endl;

    // --------------!! two threads for image warping !!-----------------------
    void** arg1 = new void*[2];
    void** arg2 = new void*[2];
    int tmp1;
    int tmp2;
    for (int item_id = 0; item_id < num_fg && layer->iter_ != layer->fg_windows_.end(); item_id += 2) {
        tmp1 = item_id;
        tmp2 = item_id + 1;
        arg1[0] = static_cast<void*>(layer);
        arg1[1] = static_cast<int*>(&tmp1);
        CHECK(!pthread_create(&(layer->thread1), NULL, PrefetchHelper<Dtype>,
          static_cast<void*>(arg1))) << "Pthread1 execution failed.";

        arg2[0] = static_cast<void*>(layer);
        arg2[1] = static_cast<int*>(&tmp2);
        CHECK(!pthread_create(&(layer->thread2), NULL, PrefetchHelper<Dtype>,
          static_cast<void*>(arg2))) << "Pthread1 execution failed.";

        CHECK(!pthread_join(layer->thread1, NULL)) << "Pthread joining failed.";
        CHECK(!pthread_join(layer->thread2, NULL)) << "Pthread joining failed.";

        layer->iter_++;
        if (layer->iter_ != layer->fg_windows_.end())
          layer->iter_++;
    }
    std::cout << "prefetch2 done!" << std::endl;
    return reinterpret_cast<void*>(NULL);
}

template <typename Dtype>
void* WindowDataFeatExtractLayerPrefetch3(void* layer_pointer) {
  WindowDataFeatExtractLayer<Dtype>* layer =
        reinterpret_cast<WindowDataFeatExtractLayer<Dtype>*>(layer_pointer);
  if (layer->iter_ != layer->fg_windows_.end()) {
    const int batch_size = layer->layer_param_.window_data_param().batch_size();
    void** arg1 = new void*[2];
    void** arg2 = new void*[2];
    int flag1 = 0, flag2 = 1;
    arg1[0] = layer_pointer;
    arg1[1] = &flag1;
    CHECK(!pthread_create(&(layer->thread1), NULL, WindowDataFeatExtractLayerPrefetch3Helper<Dtype>,
            static_cast<void*>(arg1))) << "Pthread1 execution failed.";

    arg2[0] = layer_pointer;
    arg2[1] = &flag2;
    CHECK(!pthread_create(&(layer->thread2), NULL, WindowDataFeatExtractLayerPrefetch3Helper<Dtype>,
            static_cast<void*>(arg2))) << "Pthread1 execution failed.";

    CHECK(!pthread_join(layer->thread1, NULL)) << "Pthread joining failed.";
    CHECK(!pthread_join(layer->thread2, NULL)) << "Pthread joining failed.";
    for (int i = 0; i < batch_size && layer->iter_ != layer->fg_windows_.end(); i++)
      layer->iter_++;
  }
  return reinterpret_cast<void*>(NULL);
}

template <typename Dtype>
void* WindowDataFeatExtractLayerPrefetch3Helper(void* arg) {
  void** args = (void**)arg;
  WindowDataFeatExtractLayer<Dtype>* layer =
      reinterpret_cast<WindowDataFeatExtractLayer<Dtype>*>(args[0]);
  int flag = *reinterpret_cast<int*>(args[1]);

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

  // zero out batch
  memset(top_data, 0, sizeof(Dtype)*layer->prefetch_data_->count());

  const int num_fg = static_cast<int>(batch_size);
  // const int num_samples[2] = { batch_size - num_fg, num_fg };
  int start = flag == 0 ? 0 : num_fg / 2;
  int end = flag == 0 ? num_fg / 2 : num_fg;
  vector<vector<float> >::iterator iter_ = layer->iter_;
  for (int i = 0; i < start && iter_ != layer->fg_windows_.end(); i++)
    iter_++;
  // sample from bg set then fg set
   // load the image containing the window
  // std::cout << "window num: " << layer->fg_windows_.size() << std::endl;
  // std::cout << "start, end: " << start << ", " << end << std::endl;
  for (int item_id = start; item_id < end && iter_ != layer->fg_windows_.end(); item_id++) {
    //std::cout << "item_id: " << item_id << "!!!" << std::endl;
    vector<float>  window = *(iter_);
    iter_++;
    pair<std::string, vector<int> > image =
      layer->image_database_[window[WindowDataFeatExtractLayer<Dtype>::IMAGE_INDEX]];
    cv::Mat cv_img = cv::imread(image.first, CV_LOAD_IMAGE_COLOR);
    if (!cv_img.data) {
      LOG(ERROR) << "Cound not open or find file " << image.first;
      return reinterpret_cast<void*>(NULL);
    }
    const int channels = cv_img.channels();
    // crop window out of image and warp it
    int x1 = window[WindowDataFeatExtractLayer<Dtype>::X1];
    int y1 = window[WindowDataFeatExtractLayer<Dtype>::Y1];
    int x2 = window[WindowDataFeatExtractLayer<Dtype>::X2];
    int y2 = window[WindowDataFeatExtractLayer<Dtype>::Y2];

    // validate bbox
    x1 = std::max(0, x1);
    x1 = std::min(cv_img.cols - 1, x1);
    x2 = std::max(0, x2);
    x2 = std::min(cv_img.cols - 1, x2);
    x1 = std::min(x1, x2); x2 = std::max(x1, x2);
    y1 = std::max(0, y1);
    y1 = std::min(cv_img.rows - 1, y1);
    y2 = std::max(0, y2);
    y2 = std::min(cv_img.rows - 1, y2);
    y1 = std::min(y1, y2); y2 = std::max(y1, y2);
    // std::cout << "x1, x2:" << x1 << ", " << x2 <<  std::endl;
    // std::cout << "y1, y2:" << y1 << ", " << y2 <<  std::endl;


    int pad_w = 0;
    int pad_h = 0;
    Dtype context_scale = static_cast<Dtype>(crop_size) /
        static_cast<Dtype>(crop_size - 2*context_pad);
    // compute the expanded region
    Dtype half_height = static_cast<Dtype>(y2-y1+1)/2.0;
    Dtype half_width = static_cast<Dtype>(x2-x1+1)/2.0;
    Dtype center_x = static_cast<Dtype>(x1) + half_width;
    Dtype center_y = static_cast<Dtype>(y1) + half_height;

    x1 = static_cast<int>(round(center_x - half_width*context_scale));
    x2 = static_cast<int>(round(center_x + half_width*context_scale));
    y1 = static_cast<int>(round(center_y - half_height*context_scale));
    y2 = static_cast<int>(round(center_y + half_height*context_scale));

    // the expanded region may go outside of the image
    // so we compute the clipped (expanded) region and keep track of
    // the extent beyond the image
    // std::cout << "x1, x2:" << x1 << ", " << x2 <<  std::endl;
    // std::cout << center_y << ", " << half_height << std::endl;
    // std::cout << "y1, y2:" << y1 << ", " << y2 <<  std::endl;
    int unclipped_height = y2-y1+1;
    int unclipped_width = x2-x1+1;
    int pad_x1 = std::max(0, -x1);
    int pad_y1 = std::max(0, -y1);
    // clip bounds
    x1 = std::max(0, x1);
    x2 = std::min(cv_img.cols - 1, x2);
    y1 = std::max(0, y1);
    y2 = std::min(cv_img.rows - 1, y2);
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
        // std::cout << x1 << "::" << x2 << std::endl;
        // std::cout << y1 << "::" << y2<< std::endl;
        // std::cout << cv_img.cols << ",," << cv_img.rows << std::endl;
    // size to warp the clipped expanded region to
    cv_crop_size.width =
        static_cast<int>(round(static_cast<Dtype>(clipped_width)*scale_x));
    cv_crop_size.height =
        static_cast<int>(round(static_cast<Dtype>(clipped_height)*scale_y));
    pad_x1 = static_cast<int>(round(static_cast<Dtype>(pad_x1)*scale_x));
    pad_y1 = static_cast<int>(round(static_cast<Dtype>(pad_y1)*scale_y));
    // std::cout << "crop_w, crop_h:" << cv_crop_size.width << ", " << cv_crop_size.height << std::endl;
    // std::cout << "pad_x1, scale_x:" << pad_x1 << "," << scale_x << std::endl;
    // std::cout << "pad_y1, scale_y:" << pad_y1 << "," << scale_y << std::endl;
    pad_h = pad_y1;
    pad_w = pad_x1;

    // ensure that the warped, clipped region plus the padding fits in the
    // crop_size x crop_size image (it might not due to rounding)
    if (pad_h + cv_crop_size.height > crop_size) {
      cv_crop_size.height = crop_size - pad_h;
    }
    if (pad_w + cv_crop_size.width > crop_size) {
      cv_crop_size.width = crop_size - pad_w;
    }

    cv::Rect roi(x1, y1, x2-x1+1, y2-y1+1);
    cv::Mat cv_cropped_img = cv_img(roi);
    cv::resize(cv_cropped_img, cv_cropped_img,
      cv_crop_size, 0, 0, cv::INTER_LINEAR);

    // copy the warped window into top_data
    for (int c = 0; c < channels; ++c) {
      for (int h = 0; h < cv_cropped_img.rows; ++h) {
        for (int w = 0; w < cv_cropped_img.cols; ++w) {
          Dtype pixel =
              static_cast<Dtype>(cv_cropped_img.at<cv::Vec3b>(h, w)[c]);

          top_data[((item_id * channels + c) * crop_size + h + pad_h)
                   * crop_size + w + pad_w]
              // = pixel;
              = (pixel
                  - mean[(c * mean_height + h + mean_off + pad_h)
                         * mean_width + w + mean_off + pad_w])
                * scale;
        }
      }
    }
  } // end for item_id
  // std::cout << "prefetch 3 done!" << std::endl;
  return reinterpret_cast<void*>(NULL);
}

template <typename Dtype>
void* WindowDataFeatExtractLayerPrefetch(void* layer_pointer) {
  WindowDataFeatExtractLayer<Dtype>* layer =
      reinterpret_cast<WindowDataFeatExtractLayer<Dtype>*>(layer_pointer);

  // Only do forward propagation to extract features for each iteration

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

  const int num_fg = static_cast<int>(batch_size);
  // const int num_samples[2] = { batch_size - num_fg, num_fg };

  int item_id = 0;
  // sample from bg set then fg set
   // load the image containing the window
  std::cout << "window num: " << layer->fg_windows_.size() << std::endl;
  std::cout << "num_fg: " << num_fg << std::endl;
  for (int cnt = 0; cnt < num_fg && layer->iter_ != layer->fg_windows_.end(); cnt++) {
    // std::cout << "cnt: " << cnt << "!!!" << std::endl;
    vector<float>  window = *(layer->iter_);
    layer->iter_++;
    pair<std::string, vector<int> > image =
      layer->image_database_[window[WindowDataFeatExtractLayer<Dtype>::IMAGE_INDEX]];
    cv::Mat cv_img = cv::imread(image.first, CV_LOAD_IMAGE_COLOR);
    if (!cv_img.data) {
      LOG(ERROR) << "Cound not open or find file " << image.first;
      return reinterpret_cast<void*>(NULL);
    }
    const int channels = cv_img.channels();
    // crop window out of image and warp it
    int x1 = window[WindowDataFeatExtractLayer<Dtype>::X1];
    int y1 = window[WindowDataFeatExtractLayer<Dtype>::Y1];
    int x2 = window[WindowDataFeatExtractLayer<Dtype>::X2];
    int y2 = window[WindowDataFeatExtractLayer<Dtype>::Y2];
    x1 = std::max(0, x1);
    x1 = std::min(cv_img.cols - 1, x1);
    x2 = std::max(0, x2);
    x2 = std::min(cv_img.cols - 1, x2);
    x1 = std::min(x1, x2); x2 = std::max(x1, x2);
      y1 = std::max(0, y1);
      y1= std::min(cv_img.rows - 1, y1);
      y2 = std::max(0, y2);
      y2 = std::min(cv_img.rows - 1, y2);
      y1 = std::min(y1, y2); y2 = std::max(y1, y2);
    // std::cout << "x1, x2:" << x1 << ", " << x2 <<  std::endl;
    // std::cout << "y1, y2:" << y1 << ", " << y2 <<  std::endl;


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
      // std::cout << "x1, x2:" << x1 << ", " << x2 <<  std::endl;
      // std::cout << center_y << ", " << half_height << std::endl;
      // std::cout << "y1, y2:" << y1 << ", " << y2 <<  std::endl;
      int unclipped_height = y2-y1+1;
      int unclipped_width = x2-x1+1;
      int pad_x1 = std::max(0, -x1);
      int pad_y1 = std::max(0, -y1);
      // clip bounds
      x1 = std::max(0, x1);
      x2 = std::min(cv_img.cols - 1, x2);
      y1 = std::max(0, y1);
      y2 = std::min(cv_img.rows - 1, y2);
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
          // std::cout << x1 << "::" << x2 << std::endl;
          // std::cout << y1 << "::" << y2<< std::endl;
          // std::cout << cv_img.cols << ",," << cv_img.rows << std::endl;
      // size to warp the clipped expanded region to
      cv_crop_size.width =
          static_cast<int>(round(static_cast<Dtype>(clipped_width)*scale_x));
      cv_crop_size.height =
          static_cast<int>(round(static_cast<Dtype>(clipped_height)*scale_y));
      pad_x1 = static_cast<int>(round(static_cast<Dtype>(pad_x1)*scale_x));
      pad_y1 = static_cast<int>(round(static_cast<Dtype>(pad_y1)*scale_y));
      // std::cout << "crop_w, crop_h:" << cv_crop_size.width << ", " << cv_crop_size.height << std::endl;
      // std::cout << "pad_x1, scale_x:" << pad_x1 << "," << scale_x << std::endl;
      // std::cout << "pad_y1, scale_y:" << pad_y1 << "," << scale_y << std::endl;
      pad_h = pad_y1;
      pad_w = pad_x1;

      // ensure that the warped, clipped region plus the padding fits in the
      // crop_size x crop_size image (it might not due to rounding)
      if (pad_h + cv_crop_size.height > crop_size) {
        cv_crop_size.height = crop_size - pad_h;
      }
      if (pad_w + cv_crop_size.width > crop_size) {
        cv_crop_size.width = crop_size - pad_w;
      }

      cv::Rect roi(x1, y1, x2-x1+1, y2-y1+1);
      cv::Mat cv_cropped_img = cv_img(roi);
      #if 1
      cv::resize(cv_cropped_img, cv_cropped_img,
        cv_crop_size, 0, 0, cv::INTER_LINEAR);

      // copy the warped window into top_data
      for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < cv_cropped_img.rows; ++h) {
          for (int w = 0; w < cv_cropped_img.cols; ++w) {
            Dtype pixel =
                static_cast<Dtype>(cv_cropped_img.at<cv::Vec3b>(h, w)[c]);

            top_data[((item_id * channels + c) * crop_size + h + pad_h)
                     * crop_size + w + pad_w]
                // = pixel;
                = (pixel
                    - mean[(c * mean_height + h + mean_off + pad_h)
                           * mean_width + w + mean_off + pad_w])
                  * scale;
          }
        }
      }

      // get window label (this may not be necessary)
      top_label[item_id] = window[WindowDataFeatExtractLayer<Dtype>::LABEL];
      #endif
      #if 0
      // useful debugging code for dumping transformed windows to disk
      string file_id;
      std::stringstream ss;
      ss << layer->PrefetchRand();
      ss >> file_id;
      std::ofstream inf((string("/home/common/rcnn/feat_extract/") + file_id +
          string("_info.txt")).c_str(), std::ofstream::out);
      inf << image.first << std::endl
          << window[WindowDataFeatExtractLayer<Dtype>::X1]+1 << std::endl
          << window[WindowDataFeatExtractLayer<Dtype>::Y1]+1 << std::endl
          << window[WindowDataFeatExtractLayer<Dtype>::X2]+1 << std::endl
          << window[WindowDataFeatExtractLayer<Dtype>::Y2]+1 << std::endl;
          //<< do_mirror << std::endl
          //<< top_label[item_id] << std::endl
          //<< is_fg << std::endl;
      inf.close();
      std::ofstream top_data_file((string("/home/common/rcnn/feat_extract/") + file_id +
          string("_data.txt")).c_str(),
          std::ofstream::out);
      for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < crop_size; ++h) {
          for (int w = 0; w < crop_size; ++w) {
            // top_data_file.write(reinterpret_cast<char*>(
            //     &top_data[((item_id * channels + c) * crop_size + h)
            //               * crop_size + w]),
            //     sizeof(Dtype));
            top_data_file << top_data[((item_id * channels + c) * crop_size + h)
                           * crop_size + w] << " ";
          }
          top_data_file << std::endl;
        }
      }
      // for (int c = 0; c < channels; ++c) {
      //   for (int h = 0; h < cv_cropped_img.rows; ++h) {
      //     for (int w = 0; w < cv_cropped_img.cols; ++w) {
      //       top_data_file <<
      //           static_cast<Dtype>(cv_cropped_img.at<cv::Vec3b>(h, w)[c]) << " ";
      //     }
      //     top_data_file << std::endl;
      //   }
      // }
      top_data_file.close();
      #endif
      item_id++;
    } // end if (context_pad > 0 || use_square)
  } // end for cnt
  std::cout << "prefetch done!" << std::endl;
  return reinterpret_cast<void*>(NULL);
}

template <typename Dtype>
WindowDataFeatExtractLayer<Dtype>::~WindowDataFeatExtractLayer<Dtype>() {
  JoinPrefetchThread();
}

template <typename Dtype>
void WindowDataFeatExtractLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // SetUp runs through the window_file and creates two structures
  // that hold windows: one for foreground (object) windows and one
  // for background (non-object) windows. We use an overlap threshold
  // to decide which is which.

  CHECK_EQ(bottom.size(), 0) << "Window data feature extraction Layer takes no input blobs.";
  CHECK_EQ(top->size(), 2) << "Window data feature extraction Layer prodcues two blobs as output.";

  // window_file format
  // repeated:
  //    # image_index
  //    img_path (abs path)
  //    channels
  //    height
  //    width
  //    num_windows
  //    class_index overlap x1 y1 x2 y2

  LOG(INFO) << "Window data layer:" << std::endl
      << "  foreground (object) overlap threshold: "
      << this->layer_param_.window_data_param().fg_threshold() << std::endl
      << "  background (non-object) overlap threshold: "
      << this->layer_param_.window_data_param().bg_threshold() << std::endl
      << "  foreground sampling fraction: "
      << this->layer_param_.window_data_param().fg_fraction();

  std::ifstream infile(this->layer_param_.window_data_param().source().c_str());
  CHECK(infile.good()) << "Failed to open window file "
      << this->layer_param_.window_data_param().source() << std::endl;

  map<int, int> label_hist;
  label_hist.insert(std::make_pair(0, 0));

  string hashtag;
  int image_index, channels;
  while (infile >> hashtag >> image_index) {
    CHECK_EQ(hashtag, "#");
    // read image path
    string image_path;
    infile >> image_path;
    // read image dimensions
    vector<int> image_size(3);
    infile >> image_size[0] >> image_size[1] >> image_size[2];
    channels = image_size[0];
    image_database_.push_back(std::make_pair(image_path, image_size));

    // read each box
    int num_windows;
    infile >> num_windows;
    const float fg_threshold =
        this->layer_param_.window_data_param().fg_threshold();
    const float bg_threshold =
        this->layer_param_.window_data_param().bg_threshold();
    for (int i = 0; i < num_windows; ++i) {
      int label, x1, y1, x2, y2;
      float overlap;
      infile >> label >> overlap >> x1 >> y1 >> x2 >> y2;

      vector<float> window(WindowDataFeatExtractLayer::NUM);
      window[WindowDataFeatExtractLayer::IMAGE_INDEX] = image_index;
      window[WindowDataFeatExtractLayer::LABEL] = label;
      window[WindowDataFeatExtractLayer::OVERLAP] = overlap;
      window[WindowDataFeatExtractLayer::X1] = x1;
      window[WindowDataFeatExtractLayer::Y1] = y1;
      window[WindowDataFeatExtractLayer::X2] = x2;
      window[WindowDataFeatExtractLayer::Y2] = y2;

      // add all windows to foreground list
      //if (overlap >= fg_threshold) {
        // CHECK_GT(label, 0);
        fg_windows_.push_back(window);
        label_hist.insert(std::make_pair(label, 0));
        label_hist[label]++;
      // } else if (overlap < bg_threshold) {
      //   // background window, force label and overlap to 0
      //   window[WindowDataFeatExtractLayer::LABEL] = 0;
      //   window[WindowDataFeatExtractLayer::OVERLAP] = 0;
      //   bg_windows_.push_back(window);
      //   label_hist[0]++;
      // }

    } // end for i

    if (image_index % 100 == 0) {
      LOG(INFO) << "num: " << image_index << " "
          << image_path << " "
          << image_size[0] << " "
          << image_size[1] << " "
          << image_size[2] << " "
          << "windows to process: " << num_windows;
    }
  }

  LOG(INFO) << "Number of images: " << image_index+1;
  LOG(INFO) << "Number of windows: " << fg_windows_.size();

  for (map<int, int>::iterator it = label_hist.begin();
      it != label_hist.end(); ++it) {
    LOG(INFO) << "class " << it->first << " has " << label_hist[it->first]
              << " samples";
  }

  LOG(INFO) << "Amount of context padding: "
      << this->layer_param_.window_data_param().context_pad();

  LOG(INFO) << "Crop mode: "
      << this->layer_param_.window_data_param().crop_mode();

  // image
  int crop_size = this->layer_param_.window_data_param().crop_size();
  CHECK_GT(crop_size, 0);
  const int batch_size = this->layer_param_.window_data_param().batch_size();
  (*top)[0]->Reshape(batch_size, channels, crop_size, crop_size);
  prefetch_data_.reset(
      new Blob<Dtype>(batch_size, channels, crop_size, crop_size));

  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();
  // label
  (*top)[1]->Reshape(batch_size, 1, 1, 1);
  prefetch_label_.reset(
      new Blob<Dtype>(batch_size, 1, 1, 1));

  // check if we want to have mean
  if (this->layer_param_.window_data_param().has_mean_file()) {
    const string& mean_file =
        this->layer_param_.window_data_param().mean_file();
    LOG(INFO) << "Loading mean file from" << mean_file;
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file, &blob_proto);
    data_mean_.FromProto(blob_proto);
    CHECK_EQ(data_mean_.num(), 1);
    CHECK_EQ(data_mean_.width(), data_mean_.height());
    CHECK_EQ(data_mean_.channels(), channels);
  } else {
    // Simply initialize an all-empty mean.
    data_mean_.Reshape(1, channels, crop_size, crop_size);
  }
  // Now, start the prefetch thread. Before calling prefetch, we make two
  // cpu_data calls so that the prefetch thread does not accidentally make
  // simultaneous cudaMalloc calls when the main thread is running. In some
  // GPUs this seems to cause failures if we do not so.
  prefetch_data_->mutable_cpu_data();
  prefetch_label_->mutable_cpu_data();
  data_mean_.cpu_data();
  iter_ = fg_windows_.begin();
  item_id = 0;
  DLOG(INFO) << "Initializing prefetch";
  CreatePrefetchThread();
  DLOG(INFO) << "Prefetch initialized.";
}

template <typename Dtype>
void WindowDataFeatExtractLayer<Dtype>::CreatePrefetchThread() {
  // --------------- no need for random number generator
  // const bool prefetch_needs_rand =
  //     this->layer_param_.window_data_param().mirror() ||
  //     this->layer_param_.window_data_param().crop_size();
  // if (prefetch_needs_rand) {
  //   const unsigned int prefetch_rng_seed = caffe_rng_rand();
  //   prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
  // } else {
  //   prefetch_rng_.reset();
  // }

  // Create the thread.
  CHECK(!pthread_create(&thread_, NULL, WindowDataFeatExtractLayerPrefetch3<Dtype>,
        static_cast<void*>(this))) << "Pthread execution failed.";
}

template <typename Dtype>
void WindowDataFeatExtractLayer<Dtype>::JoinPrefetchThread() {
  CHECK(!pthread_join(thread_, NULL)) << "Pthread joining failed.";
}

template <typename Dtype>
unsigned int WindowDataFeatExtractLayer<Dtype>::PrefetchRand() {
  // --------------- no need for random number generator
  // CHECK(prefetch_rng_);
  // caffe::rng_t* prefetch_rng =
  //     static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  // return (*prefetch_rng)();
  return 0;
}

template <typename Dtype>
Dtype WindowDataFeatExtractLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // First, join the thread
  JoinPrefetchThread();
  // Copy the data
  caffe_copy(prefetch_data_->count(), prefetch_data_->cpu_data(),
             (*top)[0]->mutable_cpu_data());
  caffe_copy(prefetch_label_->count(), prefetch_label_->cpu_data(),
             (*top)[1]->mutable_cpu_data());
  // Start a new prefetch thread
  CreatePrefetchThread();
  return Dtype(0.);
}

INSTANTIATE_CLASS(WindowDataFeatExtractLayer);

}  // namespace caffe
