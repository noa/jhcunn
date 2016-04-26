#include "luaT.h"
#include "THC.h"
#include "THLogAdd.h"

#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>

#include "utils.c"
#include "LogSum.cu"
#include "LogSum1d.cu"
#include "LogSample.cu"
#include "Sample.cu"
#include "LogScale.cu"
#include "Scale.cu"
#include "ClassNLLCriterionD.cu"
#include "Encode.cu"
#include "EqualSet.cu"

LUA_EXTERNC DLL_EXPORT int luaopen_libcujhu(lua_State *L);

int luaopen_libcujhu(lua_State *L) {
  lua_newtable(L);
  lua_pushvalue(L, -1);
  lua_setglobal(L, "cujhu");
  
  jhu_THCLogSum_init(L);
  jhu_THCLogSum1d_init(L);
  jhu_THCLogSample_init(L);
  jhu_THCSample_init(L);
  jhu_THCLogScale_init(L);
  jhu_THCScale_init(L);
  jhu_ClassNLLCriterionD_init(L);
  jhu_THCEncode_init(L);
  jhu_THCEqualSet_init(L);
  
  return 1;
}
