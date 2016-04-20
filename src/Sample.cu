#include "THC.h"
#include "utils.h"
#include "luaT.h"

#include <lua.h>
#include <lauxlib.h>
#include <lualib.h>

#include "THCTensorRandom.h"
#include "THCGeneral.h"
#include "THCTensorCopy.h"
#include "THCTensorMath.h"

static int jhu_THCSample(lua_State *L) {
  if (lua_gettop(L) != 2) {
    return luaL_error(L, "expecting exactly 2 arguments");
  }
  
  THCState *state = getCutorchState(L);
  
  THCudaTensor *dist = (THCudaTensor*)luaT_checkudata(L, 1,
                                                     "torch.CudaTensor");
  THCudaTensor *ret = (THCudaTensor*)luaT_checkudata(L, 2,
                                                      "torch.CudaTensor");

  THCudaTensor_multinomial(state, ret, dist, 1, false);
  
  return 0;
}

static const struct luaL_Reg jhu_THCSample__ [] = {
  {"sample", jhu_THCSample},
  {0,0}
};

static void jhu_THCSample_init(lua_State *L) {
  int ret = luaT_pushmetatable(L, "torch.CudaTensor");
  if(ret == 0) {
    THError("problem pushing metatable");
  }
  luaT_registeratname(L, jhu_THCSample__, "jhu");
  lua_pop(L, 1);
}

#undef NUM_BLOCKS
