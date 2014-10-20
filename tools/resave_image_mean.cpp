

#include <glog/logging.h>
#include <leveldb/db.h>
#include <stdint.h>

#include <algorithm>
#include <string>
#include <fstream>
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

using caffe::Datum;
using caffe::BlobProto;
//using std::max;
using namespace std;

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  if (argc != 4) {
    LOG(ERROR) << "Usage: resave_image_mean mean_txt mean_file crop_mean_file";
    return 1;
  }

  // leveldb::DB* db;
  // leveldb::Options options;
  // options.create_if_missing = false;




  
  //Datum datum;
  BlobProto sum_blob, sum_blob2;
  int count = 0;
  int w = 227;
  int full_w = 256;
  int ch=3;
  int margin = (full_w - w)/2;
  float ch_m[3] = {0.,0.,0.};
  ifstream fin;
  fin.open(argv[1]);
  float mean[3][256][256];
  LOG(INFO) << "Reading mean file " << argv[1];
  for(int ch=0; ch<3; ch++)
  for(int i = 0; i < full_w; i++)
    for(int j = 0; j < full_w; j++) {
      fin >> mean[ch][i][j];
      ch_m[ch] += mean[ch][i][j];
    }

  fin.close();
  for (int i=0; i<3; i++) {
    ch_m[i] = ch_m[i] / (full_w * full_w);
    LOG(INFO) << "channel mean " << ch_m[i];
  }
  
  const int data_size = ch * w * w;
  const int data_size2 = ch * full_w * full_w;

  sum_blob.set_num(1);
  sum_blob.set_channels(3);
  sum_blob.set_height(w);
  sum_blob.set_width(w);
  
  sum_blob2.set_num(1);
  sum_blob2.set_channels(3);
  sum_blob2.set_height(full_w);
  sum_blob2.set_width(full_w);
  for (int i = 0; i < data_size; ++i) {
    sum_blob.add_data(0.);
  }
  for (int i = 0; i < data_size2; ++i) {
    sum_blob2.add_data(0.);
  }
  for (int i = 0; i < sum_blob.data_size(); ++i) {
    int offset = i % (w * w);
    int ch = i / (w * w);
    int x = offset % w + margin;
    int y = offset / w+ margin;
    // sum_blob.set_data(i, mean[ch][y][x]);
    sum_blob.set_data(i, ch_m[ch]);
  }
  for (int i = 0; i < sum_blob2.data_size(); ++i) {
    int offset = i % (full_w * full_w);
    int ch = i / (full_w * full_w);
    int x = offset % full_w;
    int y = offset / full_w;
    // sum_blob2.set_data(i, mean[ch][y][x]);
    sum_blob2.set_data(i, ch_m[ch]);
  }
  //set to the same value (mean not related to cropping pos)
  // for (int i = 0; i < sum_blob.data_size(); ++i) {
  //   sum_blob.set_data(i, 100.);
  // }
  // for (int i = 0; i < sum_blob2.data_size(); ++i) {
  //   sum_blob2.set_data(i, 100.);
  // }
  // Write to disk
  LOG(INFO) << "Write to " << argv[2];
  WriteProtoToBinaryFile(sum_blob2, argv[2]);
  LOG(INFO) << "Write to " << argv[3];
  WriteProtoToBinaryFile(sum_blob, argv[3]);
  
  return 0;
}
