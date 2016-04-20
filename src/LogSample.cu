#include "THC.h"
#include "utils.h"
#include "luaT.h"

#include <lua.h>
#include <lauxlib.h>
#include <lualib.h>

#include "THCTensorRandom.h"
#include "THCDeviceUtils.cuh"
#include "THCReduceApplyUtils.cuh"
#include "THCApply.cuh"
#include "THCGeneral.h"
#include "THCTensorCopy.h"
#include "THCTensorMath.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>

#include <curand.h>
#include <curand_kernel.h>
#include <curand_mtgp32_host.h>
#include <curand_mtgp32dc_p_11213.h>
#include <math_constants.h>

#define MAX_NUM_BLOCKS 64
#define BLOCK_SIZE 256

__global__ void
sampleLogMultinomialOnce(float* dest,
                         long distributions,
                         int categories,
                         float* dist) {
  extern __shared__ float smem[];
  
  for (long curDist = blockIdx.x; curDist < distributions; curDist += gridDim.x) {
    
    // Each block handles one distribution
    // First pass, find the total sum of the distribution
    float sum = -CUDART_INF_F;
    for (int cat = threadIdx.x; cat < categories; cat += blockDim.x) {
      sum = device_log_add(sum, dist[curDist * categories + cat]); // 1d indexing into 2d array
    }
    
    // threadIdx.x == 0 has the sum value from this
    sum = reduceBlock(smem, blockDim.x, sum, device_log_add_functor(), -CUDART_INF_F);
    
    // Broadcast sum and sample value
    if (threadIdx.x == 0) {
      smem[0] = sum;
      smem[1] = log(dest[curDist]) + sum;
    }
    __syncthreads();
    
    sum = smem[0];
    float sample = smem[1];
    __syncthreads();
    
    if (sum == -CUDART_INF_F || sample == -CUDART_INF_F) {
      // Choose the first element
      if (threadIdx.x == 0) {
        dest[curDist] = 1;
      }
      
      continue;
    }
    
    int chunks = THCCeilDiv(categories, (int) blockDim.x);
    float prevHighProb = -CUDART_INF_F;
    
    for (int chunk = 0; chunk < chunks; ++chunk) {
      // All threads in bounds load a value
      int cat = chunk * blockDim.x + threadIdx.x;
      
      float val = cat < categories ? dist[curDist * categories + cat] : -CUDART_INF_F;
      smem[threadIdx.x] = val;
      __syncthreads();
      
      // Perform an inclusive prefix sum of the shared memory contents
      for (int offset = 1; offset < blockDim.x; offset *= 2) {
        float val = -CUDART_INF_F;
        
        if (threadIdx.x >= offset) {
          //val = smem[threadIdx.x - offset] + smem[threadIdx.x];
          val = device_log_add(smem[threadIdx.x - offset], smem[threadIdx.x]);
        }
        
        __syncthreads();
        if (threadIdx.x >= offset) {
          smem[threadIdx.x] = val;
        }
        __syncthreads();
      }
      
      // Each thread will check to see if the sample falls in its
      // bucket
      float curBucket =
        //smem[threadIdx.x] + prevHighProb;
        device_log_add(smem[threadIdx.x], prevHighProb);
      float prevBucket =
        //threadIdx.x == 0 ? prevHighProb : smem[threadIdx.x - 1] + prevHighProb;
        threadIdx.x == 0 ? prevHighProb : device_log_add(smem[threadIdx.x - 1], prevHighProb);
      bool inBucket =
        (cat < categories) && (sample <= curBucket) && (sample > prevBucket);
      
      if (inBucket) {
        // We're done; we have the sample
        // Torch indices are 1-based
        // FIXME: broadcast exit flag?
        dest[curDist] = cat + 1;
      }
      
      // Store the previous scan's high value for future use
      //prevHighProb += smem[blockDim.x - 1];
      prevHighProb = device_log_add(prevHighProb, smem[blockDim.x - 1]);
      
      __syncthreads();
    }
  }
}

void jhu_cuda_log_sample(struct THCState *state,
                         THCudaTensor *self,
                         THCudaTensor *prob_dist) {
  
  THAssert(THCudaTensor_checkGPU(state, 2, self, prob_dist));
  if (state->rngState->current_gen == NULL) {
      THError("Random number generators have not been initialized.");
  }
  
  int inputSize = THCudaTensor_nDimension(state, prob_dist);
  THArgCheck(inputSize > 0 && inputSize <= 2, 2,
             "prob_dist must be 1 or 2 dim");
  
  // Categories are in the innermost dimension
  long numDist =
    inputSize == 1 ? 1 : THCudaTensor_size(state, prob_dist, 0);
  long numCategoriesLong =
    inputSize == 1 ? THCudaTensor_size(state, prob_dist, 0) :
    THCudaTensor_size(state, prob_dist, 1);
  
  // Since the index tensor is float, numCategories cannot exceed max
  // float integer precision
  THArgCheck(numCategoriesLong <= FLOAT32_MAX_CONSECUTIVE_INT, 2,
             "number of categories cannot exceed 2^24");
  int numCategories = (int) numCategoriesLong;
  
  // It is possible that prob_dist is non-contiguous
  THCudaTensor* probDistContig =
    THCudaTensor_newContiguous(state, prob_dist);
  
  // Restructure data for 2d
  if (inputSize == 1) {
    THCudaTensor_resize2d(state, probDistContig, 1, numCategories);
  }
  
  THCudaTensor_resize2d(state, self, numDist, 1);
  
  // Optimized allocation-free implementation
  
  // To exploit greater parallelism for the sampling, generate the
  // Uniform random samples in a separate kernel launch, into the
  // result memory. The device RNG is thread-limited
  THCudaTensor_uniform(state, self, 0.0, 1.0);
  
  cudaDeviceProp* props = THCState_getCurrentDeviceProperties(state);
  THAssert(props != NULL);
  
  int numSM = props->multiProcessorCount;
  int maxThreads = props->maxThreadsPerBlock;
  
  dim3 block(numCategories < maxThreads ? numCategories : maxThreads);
  dim3 grid(numDist < numSM * 4 ? numDist : numSM * 4);
  
  sampleLogMultinomialOnce
    <<<grid, block, block.x * sizeof(float),
    THCState_getCurrentStream(state)>>>(
                                        THCudaTensor_data(state, self),
                                        numDist,
                                        numCategories,
                                        THCudaTensor_data(state, probDistContig));
  
  // Revert data restructuring based on input sizes
  if (inputSize == 1) {
    THCudaTensor_resize1d(state, self, 1);
    
    // Unfortunately, if prob_dist is contiguous already,
    // newContiguous is not a private copy, so we have to restructure
    // this too, so as to not affect prob_dist
    THCudaTensor_resize1d(state, probDistContig, numCategories);
  }
  
  THCudaTensor_free(state, probDistContig);
}

static int jhu_THCLogSample(lua_State *L) {
  THCState *state = getCutorchState(L);
  
  THCudaTensor *ret = (THCudaTensor*)luaT_checkudata(L, 2,
                                                     "torch.CudaTensor");
  THCudaTensor *dist = (THCudaTensor*)luaT_checkudata(L, 1,
                                                      "torch.CudaTensor");
  
  jhu_cuda_log_sample(state, ret, dist);
  
  return 0;
}

static const struct luaL_Reg jhu_THCLogSample__ [] = {
  {"logsample", jhu_THCLogSample},
  {0,0}
};

static void jhu_THCLogSample_init(lua_State *L) {
  int ret = luaT_pushmetatable(L, "torch.CudaTensor");
  if(ret == 0) {
    THError("problem pushing metatable");
  }
  luaT_registeratname(L, jhu_THCLogSample__, "jhu");
  lua_pop(L, 1);
}

#undef NUM_BLOCKS
