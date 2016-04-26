#include "THC.h"
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

#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>

#include <thrust/for_each.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>

struct encode_functor {
    const int n;
    encode_functor(int _n) : n(_n) {}
    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple t) {
        thrust::get<3>(t) = thrust::get<0>(t) + (thrust::get<1>(t)-1) * n;
    }
};

__global__
void encode(int N, int n, float *input0, float *input1, float *output) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < N) {
        output[i] = input0[i] + (input1[i]-1)*n;
    }
}

struct decode1_functor {
    const float a;
    decode1_functor(float _a) : a(_a) {}
    __host__ __device__ float operator()(const float& x, const float& y) const {
        return a * x + y;
    }
};

struct decode2_functor {
    const float a;
    decode2_functor(float _a) : a(_a) {}
    __host__ __device__ float operator()(const float& x, const float& y) const {
        return a * x + y;
    }
};

__global__
void decode(int N, int n, float *input, float *output0, float *output1) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < N) {
        int x = floorf(input[i] + 0.5f)-1;
        output0[i] = (x % n) + 1;
        output1[i] = (x / n) + 1;
    }
}

static int jhu_THCEncode(lua_State *L) {
    int narg = lua_gettop(L);
    if (narg != 4) {
        THError("expecting exactly 4 arguments");
    }

    THCState *state = getCutorchState(L);

    THCudaTensor *input0 = (THCudaTensor *)luaT_checkudata(L, 1,
                                                           "torch.CudaTensor");
    THCudaTensor *input1 = (THCudaTensor *)luaT_checkudata(L, 2,
                                                           "torch.CudaTensor");
    THCudaTensor *output = (THCudaTensor *)luaT_checkudata(L, 3,
                                                           "torch.CudaTensor");
    long N = lua_tonumber(L, 4);

    if(!THCudaTensor_isContiguous(state, input0) ||
       !THCudaTensor_isContiguous(state, input1) ||
       !THCudaTensor_isContiguous(state, output)) {
        THError("tensor arguments must be contiguous");
    }

    float *input_data0 = THCudaTensor_data(state, input0);
    float *input_data1 = THCudaTensor_data(state, input1);
    float *output_data = THCudaTensor_data(state, output);

    int nelem1 = THCudaTensor_size(state, input0, 0);
    int nelem2 = THCudaTensor_size(state, input1, 0);

    int ndim1 = THCudaTensor_nDimension(state, input0);
    int ndim2 = THCudaTensor_nDimension(state, input1);

    if (ndim1 != ndim2)   THError("dim mismatch");
    if (nelem1 != nelem2) THError("size mismatch");

    encode<<<(nelem1+255)/256, 256>>>(nelem1, N, input_data0, input_data1, output_data);
    
    /////////////////////////// THRUST VERSION DOESN'T COMPILE ///////////////////////////
    
    // thrust::device_ptr<float> input0_start = thrust::device_pointer_cast(input_data0);
    // thrust::device_ptr<float> input0_stop  = input0_start + nelem1;
    // thrust::device_ptr<float> input1_start = thrust::device_pointer_cast(input_data1);
    // thrust::device_ptr<float> input1_stop  = input1_start + nelem1;
    // thrust::device_ptr<float> output_start = thrust::device_pointer_cast(input_data1);
    // thrust::device_ptr<float> output_stop  = output_start + nelem1;

    // thrust::device_vector<float> ivec1(input0_start, input0_stop);
    // thrust::device_vector<float> ivec2(input1_start, input1_stop);
    // thrust::device_vector<float> ovec1(output_start, output_stop);

    // typedef thrust::device_vector<float>::iterator FloatIterator;
    // typedef thrust::tuple<FloatIterator, FloatIterator, FloatIterator> IteratorTuple;
    // typedef thrust::zip_iterator<IteratorTuple> ZipIterator;

    // // finally, create the zip_iterators
    // ZipIterator start_iter(thrust::make_tuple(ivec1.begin(), ivec2.begin(), ovec1.begin()));
    // ZipIterator end_iter(thrust::make_tuple(ivec1.end(),     ivec2.end(),   ovec1.end()));
    
    // // apply the transformation
    // thrust::for_each(start_iter, end_iter, encode_functor(N));
    
    return 0;
}

static int jhu_THCDecode(lua_State *L) {
    int narg = lua_gettop(L);
    if (narg != 4) {
        THError("expecting exactly 4 arguments");
    }

    THCState *state = getCutorchState(L);

    THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 1,
                                                          "torch.CudaTensor");
    THCudaTensor *output0 = (THCudaTensor *)luaT_checkudata(L, 2,
                                                            "torch.CudaTensor");
    THCudaTensor *output1 = (THCudaTensor *)luaT_checkudata(L, 3,
                                                            "torch.CudaTensor");
    long N = lua_tonumber(L, 4);

    float *input_data   = THCudaTensor_data(state, input);
    float *output_data0 = THCudaTensor_data(state, output0);
    float *output_data1 = THCudaTensor_data(state, output1);

    int nelem0 = THCudaTensor_size(state, input, 0);
    int nelem1 = THCudaTensor_size(state, output0, 0);
    int nelem2 = THCudaTensor_size(state, output1, 0);

    int ndim1 = THCudaTensor_nDimension(state, output0);
    int ndim2 = THCudaTensor_nDimension(state, output1);

    if (ndim1 != ndim2)   THError("dim mismatch");
    if (nelem0 != nelem1) THError("size mismatch");
    if (nelem1 != nelem2) THError("size mismatch");

    decode<<<(nelem0+255)/256, 256>>>(nelem0, N, input_data, output_data0, output_data1);
    
    return 0;
}

static const struct luaL_Reg jhu_THEncode__ [] = {
    {"encode", jhu_THCEncode},
    {"decode", jhu_THCDecode},
    {0, 0}
};

static void jhu_THCEncode_init(lua_State *L) {
    int ret = luaT_pushmetatable(L, "torch.CudaTensor");
    if(ret == 0) {
        THError("problem pushing metatable");
    }
    luaT_registeratname(L, jhu_THEncode__, "jhu");
    lua_pop(L, 1);
}
