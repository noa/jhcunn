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

__device__ float log_add(const float& a, const float& b) {
    if (a == -CUDART_INF_F) return b;
    if (b == -CUDART_INF_F) return a;
    return a>b ? a+log1p(exp(b-a)) : b+log1p(exp(a-b));
}

struct log_add_functor {
    __device__ float operator() (const float& a, const float& b) {
        return log_add(a, b);
    }
};

// convert a linear index to a row index
template <typename T>
struct linear_index_to_row_index : public thrust::unary_function<T,T> {
    T C; // number of columns
  
    __host__ __device__
    linear_index_to_row_index(T C) : C(C) {}
    
    __host__ __device__
    T operator()(T i) {
        return i / C;
    }
};

static int jhu_THCLogSum(lua_State *L) {
    THCState *state = getCutorchState(L);
    THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 1, "torch.CudaTensor");
    THCudaTensor *output = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
    
    cudaDeviceProp* props = THCState_getCurrentDeviceProperties(state);
    THAssert(props != NULL);
    
    long R, C;
    long ndim = THCudaTensor_nDimension(state, input);
    
    if(ndim == 2) {
        R = THCudaTensor_size(state, input, 0);
        C = THCudaTensor_size(state, input, 1);
        
        thrust::device_ptr<float> array  = thrust::device_pointer_cast(THCudaTensor_data(state, input));
        thrust::device_ptr<float> row_sums = thrust::device_pointer_cast(THCudaTensor_data(state, output));
        thrust::device_vector<int> row_indices(R);
    
        thrust::reduce_by_key
            (thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(C)),
             thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(C)) + (R*C),
             array,
             row_indices.begin(),
             row_sums,
             thrust::equal_to<int>(),
             log_add_functor());
    }
    else {
        THArgCheck(0, 2, "matrix expected");
    }
    
    return 0;
}

static const struct luaL_Reg jhu_THCLogSum__ [] = {
    {"logsum", jhu_THCLogSum},
    {0, 0}
};

static void jhu_THCLogSum_init(lua_State *L) {
    int ret = luaT_pushmetatable(L, "torch.CudaTensor");
    if(ret == 0) {
        THError("problem pushing metatable");
    }
    luaT_registeratname(L, jhu_THCLogSum__, "jhu");
    lua_pop(L, 1);
}
