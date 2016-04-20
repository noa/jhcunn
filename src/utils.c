#include "utils.h"

#include <math_constants.h>

THCState* getCutorchState(lua_State* L) {
  lua_getglobal(L, "cutorch");
  lua_getfield(L, -1, "getState");
  lua_call(L, 0, 1);
  THCState *state = (THCState*) lua_touserdata(L, -1);
  lua_pop(L, 2);
  return state;
}

__device__ float device_log_add(const float& a, const float& b) {
  if (a == -CUDART_INF_F) return b;
  if (b == -CUDART_INF_F) return a;
  return a>b ? a+log1p(exp(b-a)) : b+log1p(exp(a-b));
}

struct device_log_add_functor {
  __device__ float operator() (const float& a, const float& b) {
    return device_log_add(a, b);
  }
};
