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

__global__ void inplace_renorm_rows(float* dist, long rows, long cols) {
    extern __shared__ float smem[];

    for (long row = blockIdx.x; row < rows; row += gridDim.x) {
        float sum = 0;
        for (long col = threadIdx.x; col < cols; col += blockDim.x) {
            sum += dist[row * cols + col];
        }
        sum = reduceBlock(smem, blockDim.x, sum, thrust::plus<float>(), 0.0f);

        if (threadIdx.x == 0) {
            smem[0] = sum;
        }
        __syncthreads();

        sum = smem[0];

        if (sum > 0.0f) {
            for (long col = threadIdx.x; col < cols; col += blockDim.x) {
                dist[row * cols + col] = dist[row * cols + col]/sum;
            }
        }
    }
}

static int jhu_THCScale(lua_State *L) {
  THCState *state = getCutorchState(L);
  THCudaTensor *t = (THCudaTensor*)luaT_checkudata(L, 1,
                                                   "torch.CudaTensor");

  cudaDeviceProp* props = THCState_getCurrentDeviceProperties(state);
  THAssert(props != NULL);
  
  int numSM = props->multiProcessorCount;
  int maxThreads = props->maxThreadsPerBlock;
  
  int dim = THCudaTensor_nDimension(state, t);
  if(dim == 1) {
    long rows = 1;
    long cols = THCudaTensor_size(state, t, 0);
    
    dim3 grid(rows < numSM * 4 ? rows : numSM * 4);
    dim3 block(cols < maxThreads ? cols : maxThreads);
    
    inplace_renorm_rows
      <<<grid, block, block.x * sizeof(float),
      THCState_getCurrentStream(state)>>>(THCudaTensor_data(state, t),
                                          rows, cols);
  } else if(dim ==2) {
    long rows = THCudaTensor_size(state, t, 0);
    long cols = THCudaTensor_size(state, t, 1);
    
    dim3 grid(rows < numSM * 4 ? rows : numSM * 4);
    dim3 block(cols < maxThreads ? cols : maxThreads);
    
    inplace_renorm_rows
      <<<grid, block, block.x * sizeof(float),
      THCState_getCurrentStream(state)>>>(THCudaTensor_data(state, t),
                                          rows, cols);
  } else {
    THArgCheck(0, 2, "vector or matrix expected");
  }
  
  return 0;
}

static const struct luaL_Reg jhu_THCScale__ [] = {
  {"scale", jhu_THCScale},
  {0,0}
};

static void jhu_THCScale_init(lua_State *L) {
  int ret = luaT_pushmetatable(L, "torch.CudaTensor");
  if(ret == 0) {
    THError("problem pushing metatable");
  }
  luaT_registeratname(L, jhu_THCScale__, "jhu");
  lua_pop(L, 1);
}

#undef NUM_BLOCKS
