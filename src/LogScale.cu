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

__global__ void log_renorm_rows(float* dist, float* dst, long rows, long cols) {
    extern __shared__ float smem[];

    for (long row = blockIdx.x; row < rows; row += gridDim.x) {
        float sum = -CUDART_INF_F;
        for (long col = threadIdx.x; col < cols; col += blockDim.x) {
            sum = log_add(sum, dist[row * cols + col]);
        }
        sum = reduceBlock(smem, blockDim.x, sum, log_add_functor(), -CUDART_INF_F);

        if (threadIdx.x == 0) {
            smem[0] = sum;
        }
        __syncthreads();

        sum = smem[0];

        if (sum > -CUDART_INF_F) {
            for (long col = threadIdx.x; col < cols; col += blockDim.x) {
                dst[row * cols + col] = exp(dist[row * cols + col] - sum);
            }
        }
    }
}

static int jhu_THCLogScale(lua_State *L) {
    int narg = lua_gettop(L);
    
    THCState *state = getCutorchState(L);
    THCudaTensor *t = (THCudaTensor*)luaT_checkudata(L, 1,
                                                     "torch.CudaTensor");
    
    cudaDeviceProp* props = THCState_getCurrentDeviceProperties(state);
    THAssert(props != NULL);
    
    int numSM = props->multiProcessorCount;
    int maxThreads = props->maxThreadsPerBlock;
    
    int dim = THCudaTensor_nDimension(state, t);

    long rows;
    long cols;
    
    if(dim == 1) {
        rows = 1;
        cols = THCudaTensor_size(state, t, 0);
    } else if(dim ==2) {
        rows = THCudaTensor_size(state, t, 0);
        cols = THCudaTensor_size(state, t, 1);
    } else {
        THArgCheck(0, 2, "vector or matrix expected");
    }

    dim3 grid(rows < numSM * 4 ? rows : numSM * 4);
    dim3 block(cols < maxThreads ? cols : maxThreads);

    float *input = THCudaTensor_data(state, t);
    float *output;

    if (narg == 2) {
        THCudaTensor *output_tensor = (THCudaTensor*)luaT_checkudata(L, 2,
                                                                     "torch.CudaTensor");
        if(dim != THCudaTensor_nDimension(state, output_tensor)) {
            THError("ndim mismatch");
        }
        if(THCudaTensor_size(state, t, 0) != THCudaTensor_size(state, output_tensor, 0)) {
            THError("dim 0 mismatch");
        }
        output = THCudaTensor_data(state, output_tensor);
    } else if(narg == 1) {
        output = input;
    } else {
        THError("unexpected number of arguments");
    }
    
    log_renorm_rows
        <<<grid, block, block.x * sizeof(float),
        THCState_getCurrentStream(state)>>>(input,
                                            output,
                                            rows, cols);
    
    return 0;
}

static const struct luaL_Reg jhu_THCLogScale__ [] = {
    {"logscale", jhu_THCLogScale},
    {0,0}
};

static void jhu_THCLogScale_init(lua_State *L) {
  int ret = luaT_pushmetatable(L, "torch.CudaTensor");
  if(ret == 0) {
      THError("problem pushing metatable");
  }
  luaT_registeratname(L, jhu_THCLogScale__, "jhu");
  lua_pop(L, 1);
}

#undef NUM_BLOCKS
