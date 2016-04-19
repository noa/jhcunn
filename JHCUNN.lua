local ffi = require 'ffi'
local THNN = require 'nn.THNN'
local JHNN = require 'jhnn.JHNN'

print("JHNN")
print(JHNN)

local JHCUNN = {}

-- load libTHCUNN

-- This loads the dynamic library given by name and returns a new C
 --library namespace which binds to its symbols. On POSIX systems, if
 --global is true, the library symbols are loaded into the global
 --namespace, too.  If name is a path, the library is loaded from this
 --path. Otherwise name is canonicalized in a system-dependent way and
 --searched in the default search path for dynamic libraries: On POSIX
 --systems, if the name contains no dot, the extension .so is
 --appended. Also, the lib prefix is prepended if necessary. So
 --ffi.load("z") looks for "libz.so" in the default shared library
 --search path.then
JHCUNN.C = ffi.load(package.searchpath('libJHCUNN', package.cpath))

print('JHCUNN.C:')
print(JHCUNN.C)
print('---')

-- Creates a ctype object for the given ct. This function is
-- especially useful to parse a cdecl only once and then use the
-- resulting ctype object as a constructor.
local THCState_ptr = ffi.typeof('THCState*')

print('THCState_ptr = ')
print(THCState_ptr)

function JHCUNN.getState()
   return THCState_ptr(cutorch.getState())
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

print('metatable:')
print(torch.getmetatable('torch.CudaTensor').JHNN)

return JHCUNN
