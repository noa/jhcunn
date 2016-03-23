local ffi = require 'ffi'
local JHNN = require 'nn.JHNN'

local JHCUNN = {}

-- load libTHCUNN
JHCUNN.C = ffi.load(package.searchpath('libJHCUNN', package.cpath))

local THCState_ptr = ffi.typeof('THCState*')

function JHCUNN.getState()
   return THCState_ptr(cutorch.getState());
end

local JHCUNN_h = require 'cunn.JHCUNN_h'
-- strip all lines starting with #
-- to remove preprocessor directives originally present
-- in THNN.h
JHCUNN_h = JHCUNN_h:gsub("\n#[^\n]*", "")
JHCUNN_h = JHCUNN_h:gsub("^#[^\n]*\n", "")

local preprocessed = string.gsub(JHCUNN_h, 'TH_API ', '')

local replacements =
{
   {
      ['THTensor'] = 'THCudaTensor',
      ['THIndexTensor'] = 'THCudaTensor',
      ['THIntegerTensor'] = 'THCudaTensor',
      ['THIndex_t'] = 'float',
      ['THInteger_t'] = 'float'
   }
}

for i=1,#replacements do
   local r = replacements[i]
   local s = preprocessed
   for k,v in pairs(r) do
      s = string.gsub(s, k, v)
   end
   ffi.cdef(s)
end

local function extract_function_names(s)
   local t = {}
   for n in string.gmatch(s, 'TH_API void JHNN_Cuda([%a%d_]+)') do
      t[#t+1] = n
   end
   return t
end

-- build function table
local function_names = extract_function_names(JHCUNN_h)

JHNN.kernels['torch.CudaTensor'] = JHNN.bind(JHCUNN.C, function_names, 'Cuda', JHCUNN.getState)
torch.getmetatable('torch.CudaTensor').THNN = JHNN.kernels['torch.CudaTensor']

return JHCUNN
