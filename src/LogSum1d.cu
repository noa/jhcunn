#include "THC.h"
#include "utils.h"
#include "luaT.h"

#include <lua.h>
#include <lauxlib.h>
#include <lualib.h>

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
#include <math_constants.h>

struct LogAddFunctor : public thrust::binary_function<float,float,float> {
    const float NEGATIVE_INFINITY;
    LogAddFunctor(float _NEGATIVE_INFINITY) : NEGATIVE_INFINITY(_NEGATIVE_INFINITY) {}
    __host__ __device__ float operator() (float a, float b) const {
        if (a == NEGATIVE_INFINITY) return b;
        if (b == NEGATIVE_INFINITY) return a;
        return a>b ? a+log1p(exp(b-a)) : b+log1p(exp(a-b));
    }
};


static int jhu_THCLogSum1d(lua_State *L) {
    THCState *state = getCutorchState(L);
    THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 1, "torch.CudaTensor");
    
    cudaDeviceProp* props = THCState_getCurrentDeviceProperties(state);
    THAssert(props != NULL);
    
    long ndim = THCudaTensor_nDimension(state, input);

    if(ndim != 1) {
        THError("input must be 1d");
    }

    const float NEGATIVE_INFINITY = -std::numeric_limits<float>::infinity();
    
    long N = THCudaTensor_size(state, input, 0);
    thrust::device_ptr<float> array = thrust::device_pointer_cast(THCudaTensor_data(state, input));
    float result = thrust::reduce(array, array+N,
                                  NEGATIVE_INFINITY,
                                  LogAddFunctor(NEGATIVE_INFINITY));

    lua_pushnumber(L, result);
    
    return 1;
}

static const struct luaL_Reg jhu_THCLogSum1d__ [] = {
    {"logsum1d", jhu_THCLogSum1d},
    {0, 0}
};

static void jhu_THCLogSum1d_init(lua_State *L) {
    int ret = luaT_pushmetatable(L, "torch.CudaTensor");
    if(ret == 0) {
        THError("problem pushing metatable");
    }
    luaT_registeratname(L, jhu_THCLogSum1d__, "jhu");
    lua_pop(L, 1);
}
