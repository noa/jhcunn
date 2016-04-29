#include "THC.h"
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

__global__
void equalset1(int N, float *input, float *output, int val1, int val2) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < N) {
        if((int)input[i] == val1) {
            output[i] = input[i];
        } else {
            output[i] = (float)val2;
        }
    }
}

__global__
void equalset2(int N, float *input, float *output, int val2) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < N) {
        if((int)input[i] == (int)output[i]) {
            output[i] = input[i];
        } else {
            output[i] = (float)val2;
        }
    }
}

static int jhu_THCEqualSet(lua_State *L) {
    int narg = lua_gettop(L);
    if (narg != 4) {
        THError("expecting exactly 4 arguments");
    }

    THCState *state = getCutorchState(L);

    THCudaTensor *output = (THCudaTensor *)luaT_checkudata(L, 1,
                                                           "torch.CudaTensor");
    THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2,
                                                          "torch.CudaTensor");
    int val1 = lua_tonumber(L, 3);
    int val2 = lua_tonumber(L, 4);

    if(!THCudaTensor_isContiguous(state, input) ||
       !THCudaTensor_isContiguous(state, output)) {
        THError("tensor arguments must be contiguous");
    }

    int size1 = THCudaTensor_size(state, input, 0);
    if(!(size1 == THCudaTensor_size(state, output, 0))) THError("size mismatch");
    if(!(THCudaTensor_nDimension(state, output) ==
         THCudaTensor_nDimension(state, input)
           )) THError("dim mismatch");

    float *input_data = THCudaTensor_data(state, input);
    float *output_data = THCudaTensor_data(state, output);

    int ndim1 = THCudaTensor_nDimension(state, input);
    int ndim2 = THCudaTensor_nDimension(state, output);

    if (ndim1 != ndim2) THError("dim mismatch");
    if (val1 == 0) THError("val should be 1-indexed or -1 if unused");
    
    if(val1 > 0) {
        equalset1<<<(size1+255)/256, 256>>>(size1, input_data, output_data, val1, val2);
    } else {
        equalset2<<<(size1+255)/256, 256>>>(size1, input_data, output_data, val2);
    }
        
    
    return 0;
}

static const struct luaL_Reg jhu_THCEqualSet__ [] = {
    {"equalset", jhu_THCEqualSet},
    {0, 0}
};

static void jhu_THCEqualSet_init(lua_State *L) {
    int ret = luaT_pushmetatable(L, "torch.CudaTensor");
    if(ret == 0) {
        THError("problem pushing metatable");
    }
    luaT_registeratname(L, jhu_THCEqualSet__, "jhu");
    lua_pop(L, 1);
}
