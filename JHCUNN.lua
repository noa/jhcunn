local ffi = require 'ffi'
local JHNN = require 'jhnn.JHNN'

local JHCUNN = {}

-- load libTHCUNN
JHCUNN.C = ffi.load(package.searchpath('libJHCUNN', package.cpath))

local THCState_ptr = ffi.typeof('THCState*')

function JHCUNN.getState()
   return THCState_ptr(cutorch.getState());
end

local JHCUNN_h = require 'jhcunn.JHCUNN_h'
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
   print(s)
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

print('[JHCUNN] function names:')
print(function_names)

JHNN.kernels['torch.CudaTensor'] = JHNN.bind(JHCUNN.C, function_names, 'Cuda', JHCUNN.getState)
torch.getmetatable('torch.CudaTensor').JHNN = JHNN.kernels['torch.CudaTensor']

return JHCUNN
