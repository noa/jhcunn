#ifndef JHU_UTILS_H
#define JHU_UTILS_H

#include <lua.h>
#include "THCGeneral.h"

THCState* getCutorchState(lua_State* L);
__device__ float device_log_add(const float& a, const float& b);
struct device_log_add_functor;

#endif
