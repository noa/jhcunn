#include <THC/THC.h>
#include <THC/THCApply.cuh>

#define THIndexTensor THCudaTensor
#define THIndexTensor_(NAME) THCudaTensor_ ## NAME

#define THIntegerTensor THCudaTensor
#define THIntegerTensor_(NAME) THCudaTensor_ ## NAME

TH_API void JHNN_CudaLookupTable_accGradParameters(THCState *state,
                                                   THIndexTensor *input,
                                                   THCudaTensor *gradOutput,
                                                   THCudaTensor *gradWeight,
                                                   THIntegerTensor *count,
                                                   THCudaTensor *sorted,
                                                   THCudaTensor *indices,
                                                   bool scaleGradByFreq,
                                                   int paddingValue,
                                                   THCudaTensor *scale);
