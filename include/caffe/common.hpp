// Copyright 2014 BVLC and contributors.

#ifndef CAFFE_COMMON_HPP_
#define CAFFE_COMMON_HPP_

#include <boost/shared_ptr.hpp>
#include <cublas_v2.h>
#include <cuda.h>
#include <curand.h>
#include <driver_types.h>  // cuda driver types
#include <glog/logging.h>

// Disable the copy and assignment operator for a class.
#define DISABLE_COPY_AND_ASSIGN(classname) \
private:\
  classname(const classname&);\
  classname& operator=(const classname&)

// Instantiate a class with float and double specifications.
#define INSTANTIATE_CLASS(classname) \
  template class classname<float>; \
  template class classname<double>

// A simple macro to mark codes that are not implemented, so that when the code
// is executed we will see a fatal log.
#define NOT_IMPLEMENTED LOG(FATAL) << "Not Implemented Yet"

// CUDA: various checks for different function calls.
#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)

#define CUBLAS_CHECK(condition) \
  do { \
    cublasStatus_t status = condition; \
    CHECK_EQ(status, CUBLAS_STATUS_SUCCESS) << " " \
      << caffe::cublasGetErrorString(status); \
  } while (0)

#define CURAND_CHECK(condition) \
  do { \
    curandStatus_t status = condition; \
    CHECK_EQ(status, CURAND_STATUS_SUCCESS) << " " \
      << caffe::curandGetErrorString(status); \
  } while (0)

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

// CUDA: check for error after kernel execution and exit loudly if there is one.
#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())

// Define not supported status for pre-6.0 compatibility.
#if CUDA_VERSION < 6000
#define CUBLAS_STATUS_NOT_SUPPORTED 831486
#endif

namespace caffe {

// We will use the boost shared_ptr instead of the new C++11 one mainly
// because cuda does not work (at least now) well with C++11 features.
using boost::shared_ptr;


// A singleton class to hold common caffe stuff, such as the handler that
// caffe is going to use for cublas, curand, etc.
class Caffe {
 public:
  ~Caffe();
  inline static Caffe& Get() {
    if (!singleton_.get()) {
      singleton_.reset(new Caffe());
    }
    return *singleton_;
  }
  enum Brew { CPU, GPU };
  enum Phase { TRAIN, TEST };
  enum GPU_MODE { SINGLE, MASTER_SLAVE, PARALLEL};


  // This random number generator facade hides boost and CUDA rng
  // implementation from one another (for cross-platform compatibility).
  class RNG {
   public:
    RNG();
    explicit RNG(unsigned int seed);
    explicit RNG(const RNG&);
    RNG& operator=(const RNG&);
    void* generator();
   private:
    class Generator;
    shared_ptr<Generator> generator_;
  };

  // Getters for boost rng, curand, and cublas handles
  inline static RNG& rng_stream() {
    if (!Get().random_generator_) {
      Get().random_generator_.reset(new RNG());
    }
    return *(Get().random_generator_);
  }
  inline static cublasHandle_t cublas_handle() { return Get().cublas_handle_; }
  inline static curandGenerator_t curand_generator() {
    return Get().curand_generator_;
  }
  inline static cudaStream_t cu_stream() { return Get().cu_stream_; }

  inline static cublasHandle_t slave_cublas_handle() { return Get().slave_cublas_handle_; }
  inline static curandGenerator_t slave_curand_generator() {
    return Get().slave_curand_generator_;
  }
  inline static cudaStream_t slave_cu_stream() { return Get().slave_cu_stream_; }

  inline static int master_device_id() {return Get().master_device_id_;}
  inline static int slave_device_id() {return Get().slave_device_id_;}

  //swtich between master and slave device
  inline static void switch_to_master_device(){
	  Get().current_device_id_ = Get().master_device_id_;
	  CUDA_CHECK(cudaSetDevice(Get().master_device_id_));
	  cublasSetStream(Get().cublas_handle_, Get().cu_stream_);
	  Get().current_cu_stream_ = Get().cu_stream_;
  }
  inline static void switch_to_slave_device(){
	Get().current_device_id_ = Get().slave_device_id_;
	CUDA_CHECK(cudaSetDevice(Get().slave_device_id_));
	cublasSetStream(Get().slave_cublas_handle_, Get().slave_cu_stream_);
	Get().current_cu_stream_ = Get().slave_cu_stream_;
  }

  inline static void set_cu_stream(cudaStream_t& new_stream){
	Get().current_cu_stream_ = new_stream;
	if (Get().current_device_id_ == Get().master_device_id_){
		cublasSetStream(Get().cublas_handle_, new_stream);
	}
	else{
		cublasSetStream(Get().slave_cublas_handle_, new_stream);
	}
  }

  //get device info
  inline static int get_current_device_id(){ return Get().current_device_id_;}
  inline static cublasHandle_t get_current_cublas_handle(){ 
	  if ((Get().gpu_mode_ == MASTER_SLAVE)&&(Get().current_device_id_ == Get().slave_device_id_))
		  return Get().slave_cublas_handle_;
	  else
		  return Get().cublas_handle_;
  }
  inline static curandGenerator_t get_current_curand_generator(){ 
	  if ((Get().gpu_mode_ == MASTER_SLAVE)&&(Get().current_device_id_ == Get().slave_device_id_))
		  return Get().slave_curand_generator_;
	  else
		  return Get().curand_generator_;
  }
  inline static cudaStream_t get_current_cu_stream(){ 
	  /*if ((Get().gpu_mode_ == MASTER_SLAVE)&&(Get().current_device_id_ == Get().slave_device_id_))
		  return Get().slave_cu_stream_;
	  else
		  return Get().cu_stream_;*/
	  return Get().current_cu_stream_;
  }


  // Returns the mode: running on CPU or GPU.
  inline static Brew mode() { 
	  int m= Get().mode_; 
	  return Brew(m);
  }
  // Returns the phase: TRAIN or TEST.
  inline static Phase phase() { return Get().phase_; }
  // Returns GPU mode
  inline static GPU_MODE gpu_mode() {return Get().gpu_mode_;}
  // The setters for the variables
  // Sets the mode. It is recommended that you don't change the mode halfway
  // into the program since that may cause allocation of pinned memory being
  // freed in a non-pinned way, which may cause problems - I haven't verified
  // it personally but better to note it here in the header file.
  inline static void set_mode(Brew mode) { Get().mode_ = mode; }
  
  // Sets the phase.
  inline static void set_phase(Phase phase) { Get().phase_ = phase; }
  // Sets the random seed of both boost and curand
  static void set_random_seed(const unsigned int seed);

  //set gpu mode
  inline static void set_gpu_mode(GPU_MODE gpu_mode){ Get().gpu_mode_ = gpu_mode;}
  // Sets the device. Since we have cublas and curand stuff, set device also
  // requires us to reset those values.
  static void SetDevice(const int device_id);
  static void SetSlaveDevice(const int slave_device_id);
  // Prints the current GPU status.
  static void DeviceQuery();
  static void SlaveDeviceQuery(const int slave_device_id);

  static void ConnectMasterSlaveDevice(const int master_device_id, const int slave_device_id);

 protected:
  cublasHandle_t cublas_handle_;
  curandGenerator_t curand_generator_;
  shared_ptr<RNG> random_generator_;
  cudaStream_t cu_stream_;

  cublasHandle_t slave_cublas_handle_;
  curandGenerator_t slave_curand_generator_;
  shared_ptr<RNG> slave_random_generator_;
  cudaStream_t slave_cu_stream_;

  cudaStream_t current_cu_stream_;

  //vector<cudaStream_t> cuda_streams;

  Brew mode_;
  Phase phase_;
  GPU_MODE gpu_mode_;

  int master_device_id_;
  int slave_device_id_;
  int current_device_id_;
  static shared_ptr<Caffe> singleton_;

 private:
  // The private constructor to avoid duplicate instantiation.
  Caffe();

  DISABLE_COPY_AND_ASSIGN(Caffe);
};

// NVIDIA_CUDA-5.5_Samples/common/inc/helper_cuda.h
const char* cublasGetErrorString(cublasStatus_t error);
const char* curandGetErrorString(curandStatus_t error);

// CUDA: thread number configuration.
// Use 1024 threads per block, which requires cuda sm_2x or above,
// or fall back to attempt compatibility (best of luck to you).
#if __CUDA_ARCH__ >= 200
    const int CAFFE_CUDA_NUM_THREADS = 1024;
#else
    const int CAFFE_CUDA_NUM_THREADS = 512;
#endif

// CUDA: number of blocks for threads.
inline int CAFFE_GET_BLOCKS(const int N) {
  return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
}


}  // namespace caffe

#endif  // CAFFE_COMMON_HPP_
