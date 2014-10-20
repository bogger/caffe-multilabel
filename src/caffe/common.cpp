// Copyright 2014 BVLC and contributors.

#include <cstdio>
#include <ctime>

#include "caffe/common.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

shared_ptr<Caffe> Caffe::singleton_;


// curand seeding
int64_t cluster_seedgen(void) {
  int64_t s, seed, pid;
  pid = getpid();
  s = time(NULL);
  seed = abs(((s * 181) * ((pid - 83) * 359)) % 104729);
  return seed;
}


Caffe::Caffe()
    : mode_(Caffe::CPU), phase_(Caffe::TRAIN), cublas_handle_(NULL),
      curand_generator_(NULL),
      random_generator_(),
	  slave_cublas_handle_(NULL),
	  slave_curand_generator_(NULL),
	  master_device_id_(0), slave_device_id_(-1),
	  cu_stream_(NULL),slave_cu_stream_(NULL),
	  current_cu_stream_(NULL){
  // Try to create a cublas handler, and report an error if failed (but we will
  // keep the program running as one might just want to run CPU code).
  if (cublasCreate(&cublas_handle_) != CUBLAS_STATUS_SUCCESS) {
    LOG(ERROR) << "Cannot create Cublas handle. Cublas won't be available.";
  }
  // Try to create a curand handler.
  if (curandCreateGenerator(&curand_generator_, CURAND_RNG_PSEUDO_DEFAULT)
      != CURAND_STATUS_SUCCESS ||
      curandSetPseudoRandomGeneratorSeed(curand_generator_, cluster_seedgen())
      != CURAND_STATUS_SUCCESS) {
    LOG(ERROR) << "Cannot create Curand generator. Curand won't be available.";
  }


}

Caffe::~Caffe() {
  if (cublas_handle_) CUBLAS_CHECK(cublasDestroy(cublas_handle_));
  if (curand_generator_) {
    CURAND_CHECK(curandDestroyGenerator(curand_generator_));
  }
}

void Caffe::set_random_seed(const unsigned int seed) {
  // Curand seed
  // Yangqing's note: simply setting the generator seed does not seem to
  // work on the tesla K20s, so I wrote the ugly reset thing below.
  if (Get().curand_generator_) {
    CURAND_CHECK(curandDestroyGenerator(curand_generator()));
    CURAND_CHECK(curandCreateGenerator(&Get().curand_generator_,
        CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(curand_generator(),
        seed));
  } else {
    LOG(ERROR) << "Curand not available. Skipping setting the curand seed.";
  }
  // RNG seed
  Get().random_generator_.reset(new RNG(seed));
}

void Caffe::SetDevice(const int device_id) {
  int current_device;
  CUDA_CHECK(cudaGetDevice(&current_device));
  if (current_device == device_id) {
    return;
  }
  if (Get().cublas_handle_) CUBLAS_CHECK(cublasDestroy(Get().cublas_handle_));
  if (Get().curand_generator_) {
    CURAND_CHECK(curandDestroyGenerator(Get().curand_generator_));
  }
  CUDA_CHECK(cudaSetDevice(device_id));
  CUDA_CHECK(cudaStreamCreate (&Get().cu_stream_));
  CUBLAS_CHECK(cublasCreate(&Get().cublas_handle_));
  CUBLAS_CHECK(cublasSetStream(Get().cublas_handle_, Get().cu_stream_));
  
  CURAND_CHECK(curandCreateGenerator(&Get().curand_generator_,
      CURAND_RNG_PSEUDO_DEFAULT));
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(Get().curand_generator_,
      cluster_seedgen()));
  Get().master_device_id_ = device_id;
  Caffe::set_gpu_mode(Caffe::SINGLE);
}

void Caffe::SetSlaveDevice(const int slave_device_id) {
  int current_device;
  CUDA_CHECK(cudaGetDevice(&current_device));
  if (current_device == slave_device_id) {
    return;
  }
  if (Get().slave_cublas_handle_) CUBLAS_CHECK(cublasDestroy(Get().slave_cublas_handle_));
  if (Get().slave_curand_generator_) {
    CURAND_CHECK(curandDestroyGenerator(Get().slave_curand_generator_));
  }
  CUDA_CHECK(cudaSetDevice(slave_device_id));
  CUDA_CHECK(cudaStreamCreate (&Get().slave_cu_stream_));
  CUBLAS_CHECK(cublasCreate(&Get().slave_cublas_handle_));
  CUBLAS_CHECK(cublasSetStream(Get().slave_cublas_handle_, Get().slave_cu_stream_));
  CURAND_CHECK(curandCreateGenerator(&Get().slave_curand_generator_,
      CURAND_RNG_PSEUDO_DEFAULT));
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(Get().slave_curand_generator_,
      cluster_seedgen()));
  Get().slave_device_id_ = slave_device_id;
  CUDA_CHECK(cudaSetDevice(current_device));
  Caffe::set_gpu_mode(Caffe::MASTER_SLAVE);
}

void Caffe::DeviceQuery() {
  cudaDeviceProp prop;
  int device;
  if (cudaSuccess != cudaGetDevice(&device)) {
    printf("No cuda device present.\n");
    return;
  }
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  printf("Device id:                     %d\n", device);
  printf("Major revision number:         %d\n", prop.major);
  printf("Minor revision number:         %d\n", prop.minor);
  printf("Name:                          %s\n", prop.name);
  printf("Total global memory:           %lu\n", prop.totalGlobalMem);
  printf("Total shared memory per block: %lu\n", prop.sharedMemPerBlock);
  printf("Total registers per block:     %d\n", prop.regsPerBlock);
  printf("Warp size:                     %d\n", prop.warpSize);
  printf("Maximum memory pitch:          %lu\n", prop.memPitch);
  printf("Maximum threads per block:     %d\n", prop.maxThreadsPerBlock);
  printf("Maximum dimension of block:    %d, %d, %d\n",
      prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
  printf("Maximum dimension of grid:     %d, %d, %d\n",
      prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
  printf("Clock rate:                    %d\n", prop.clockRate);
  printf("Total constant memory:         %lu\n", prop.totalConstMem);
  printf("Texture alignment:             %lu\n", prop.textureAlignment);
  printf("Concurrent copy and execution: %s\n",
      (prop.deviceOverlap ? "Yes" : "No"));
  printf("Number of multiprocessors:     %d\n", prop.multiProcessorCount);
  printf("Kernel execution timeout:      %s\n",
      (prop.kernelExecTimeoutEnabled ? "Yes" : "No"));
  return;
}

void Caffe::SlaveDeviceQuery(const int slave_device_id) {
  cudaDeviceProp prop;
  int device;
  /*if (cudaSuccess != cudaGetDevice(&device)) {
    printf("No cuda device present.\n");
    return;
  }*/
  CUDA_CHECK(cudaGetDeviceProperties(&prop, slave_device_id));
  printf("Device id:                     %d\n", slave_device_id);
  printf("Major revision number:         %d\n", prop.major);
  printf("Minor revision number:         %d\n", prop.minor);
  printf("Name:                          %s\n", prop.name);
  printf("Total global memory:           %lu\n", prop.totalGlobalMem);
  printf("Total shared memory per block: %lu\n", prop.sharedMemPerBlock);
  printf("Total registers per block:     %d\n", prop.regsPerBlock);
  printf("Warp size:                     %d\n", prop.warpSize);
  printf("Maximum memory pitch:          %lu\n", prop.memPitch);
  printf("Maximum threads per block:     %d\n", prop.maxThreadsPerBlock);
  printf("Maximum dimension of block:    %d, %d, %d\n",
      prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
  printf("Maximum dimension of grid:     %d, %d, %d\n",
      prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
  printf("Clock rate:                    %d\n", prop.clockRate);
  printf("Total constant memory:         %lu\n", prop.totalConstMem);
  printf("Texture alignment:             %lu\n", prop.textureAlignment);
  printf("Concurrent copy and execution: %s\n",
      (prop.deviceOverlap ? "Yes" : "No"));
  printf("Number of multiprocessors:     %d\n", prop.multiProcessorCount);
  printf("Kernel execution timeout:      %s\n",
      (prop.kernelExecTimeoutEnabled ? "Yes" : "No"));
  return;
}

void Caffe::ConnectMasterSlaveDevice(const int master_device_id, const int slave_device_id){


	int can_access;
	int target_device_id;
	target_device_id = slave_device_id;	

	CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, master_device_id, slave_device_id));

	if (can_access == 0){
		LOG(WARNING)<<"Device P2P access from GPU "<<master_device_id<<" to GPU "<<slave_device_id<<" can not be enabled. Data transfering may be slow.";
	}else{
		CUDA_CHECK(cudaSetDevice(master_device_id));
		CUDA_CHECK(cudaDeviceEnablePeerAccess(target_device_id, 0));
		LOG(INFO)<<"Device P2P access from GPU "<<master_device_id<<" to GPU "<<slave_device_id<<"  enabled.";		
	}

	CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, slave_device_id, master_device_id ));

	if (can_access == 0){
		LOG(WARNING)<<"Device P2P access from GPU "<<slave_device_id<<" to GPU "<<master_device_id<<" can not be enabled. Data transfering may be slow.";
	}else{
		CUDA_CHECK(cudaSetDevice(slave_device_id));
		CUDA_CHECK(cudaDeviceEnablePeerAccess(master_device_id, 0));
		LOG(INFO)<<"Device P2P access from GPU "<<slave_device_id<<" to GPU "<<master_device_id<<" enabled.";
	}
	CUDA_CHECK(cudaSetDevice(master_device_id));
}


class Caffe::RNG::Generator {
 public:
  Generator() : rng_(new caffe::rng_t(cluster_seedgen())) {}
  explicit Generator(unsigned int seed) : rng_(new caffe::rng_t(seed)) {}
  caffe::rng_t* rng() { return rng_.get(); }
 private:
  shared_ptr<caffe::rng_t> rng_;
};

Caffe::RNG::RNG() : generator_(new Generator) { }

Caffe::RNG::RNG(unsigned int seed) : generator_(new Generator(seed)) { }

Caffe::RNG& Caffe::RNG::operator=(const RNG& other) {
  generator_.reset(other.generator_.get());
  return *this;
}

void* Caffe::RNG::generator() {
  return static_cast<void*>(generator_->rng());
}

const char* cublasGetErrorString(cublasStatus_t error) {
  switch (error) {
  case CUBLAS_STATUS_SUCCESS:
    return "CUBLAS_STATUS_SUCCESS";
  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "CUBLAS_STATUS_NOT_INITIALIZED";
  case CUBLAS_STATUS_ALLOC_FAILED:
    return "CUBLAS_STATUS_ALLOC_FAILED";
  case CUBLAS_STATUS_INVALID_VALUE:
    return "CUBLAS_STATUS_INVALID_VALUE";
  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "CUBLAS_STATUS_ARCH_MISMATCH";
  case CUBLAS_STATUS_MAPPING_ERROR:
    return "CUBLAS_STATUS_MAPPING_ERROR";
  case CUBLAS_STATUS_EXECUTION_FAILED:
    return "CUBLAS_STATUS_EXECUTION_FAILED";
  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "CUBLAS_STATUS_INTERNAL_ERROR";
  case CUBLAS_STATUS_NOT_SUPPORTED:
    return "CUBLAS_STATUS_NOT_SUPPORTED";
  }
  return "Unknown cublas status";
}

const char* curandGetErrorString(curandStatus_t error) {
  switch (error) {
  case CURAND_STATUS_SUCCESS:
    return "CURAND_STATUS_SUCCESS";
  case CURAND_STATUS_VERSION_MISMATCH:
    return "CURAND_STATUS_VERSION_MISMATCH";
  case CURAND_STATUS_NOT_INITIALIZED:
    return "CURAND_STATUS_NOT_INITIALIZED";
  case CURAND_STATUS_ALLOCATION_FAILED:
    return "CURAND_STATUS_ALLOCATION_FAILED";
  case CURAND_STATUS_TYPE_ERROR:
    return "CURAND_STATUS_TYPE_ERROR";
  case CURAND_STATUS_OUT_OF_RANGE:
    return "CURAND_STATUS_OUT_OF_RANGE";
  case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
    return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
  case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
    return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
  case CURAND_STATUS_LAUNCH_FAILURE:
    return "CURAND_STATUS_LAUNCH_FAILURE";
  case CURAND_STATUS_PREEXISTING_FAILURE:
    return "CURAND_STATUS_PREEXISTING_FAILURE";
  case CURAND_STATUS_INITIALIZATION_FAILED:
    return "CURAND_STATUS_INITIALIZATION_FAILED";
  case CURAND_STATUS_ARCH_MISMATCH:
    return "CURAND_STATUS_ARCH_MISMATCH";
  case CURAND_STATUS_INTERNAL_ERROR:
    return "CURAND_STATUS_INTERNAL_ERROR";
  }
  return "Unknown curand status";
}

}  // namespace caffe
