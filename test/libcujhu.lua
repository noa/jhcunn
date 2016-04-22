--luacheck: globals torch

-- Tester:
unpack = unpack or table.unpack
require 'torch'
require 'cutorch'
require 'libcujhu'

local mytest = torch.TestSuite()
local tester = torch.Tester()

function mytest.LogSum()
   -- 2D case (d1=batch, d2=dim)
   local D = 4
   local input  = torch.DoubleTensor(D,D):normal(0, 1):cuda()
   local inputCopy = input:clone()
   local output = torch.DoubleTensor(D):cuda()
   input.jhu.logsum(input, output)
   tester:eq(input, inputCopy, 1e-4)
   input:exp()
   output:exp()
   for b = 1, D do
      local sum1 = input[b]:sum()
      local sum2 = output[b]
      local diff = sum1-sum2
      tester:assert(math.abs(diff) < 1e-3, 'bad log sum: err='..diff)
   end
end

function mytest.LogSample1D()
   local D = 10
   local N = 50000
   local P = torch.DoubleTensor(D):uniform(0, 1)
   local N1 = torch.multinomial(P, N, true)
   local S1 = N1:double():sum() / N
   local logP = P:log():cuda()
   local tmp = torch.zeros(1):double():cuda()
   local N2 = torch.zeros(1):double():cuda()
   for n = 1, N do
      tmp.jhu.logsample(logP:clone(), tmp)
      N2:add(tmp)
   end
   local S2 = N2[1] / N
   local diff = math.abs(S1-S2)
   tester:assert(diff < 1e-1, 'bad log sum: err='..diff)
end

function mytest.LogSampleEdgeCase()
   local lnP = torch.CudaTensor({-math.huge, -math.huge, -math.huge, -math.huge, 0})
   local result = torch.CudaTensor(1)
   result.jhu.logsample(lnP, result)
   tester:assert(result[1] == 5, "error with edge case")
end

function mytest.LogSampleEdgeCaseTwo()
   local x = torch.Tensor({-math.huge, -math.huge, -math.huge, -0.5}):cuda()
   local y = torch.Tensor(1, 1):cuda()
   local x = x:view(1, 4)
   x.jhu.logsum(x, y)
   local z = y[1][1]
   assert(type(z) == 'number')
   assert(not (z ~= z), "NaN result!") -- check not NaN
end

function mytest.Sample1D()
   local D = 10
   local N = 50000
   local P = torch.DoubleTensor(D):uniform(0, 1)
   local N1 = torch.multinomial(P, N, true)
   local S1 = N1:double():sum() / N
   local tmp = torch.zeros(1):double():cuda()
   local N2 = torch.zeros(1):double():cuda()
   P = P:cuda()
   for n = 1, N do
      tmp.jhu.sample(P:clone(), tmp)
      N2:add(tmp)
   end
   local S2 = N2[1] / N
   local diff = math.abs(S1-S2)
   tester:assert(diff < 1e-1, 'bad sum: err='..diff)
end

function mytest.LogSampleNormalized()
   local D = 5
   local N = 50000

   local P = torch.DoubleTensor(D, D):uniform(0, 1)
   for d = 1, D do
      local Z = P[d]:sum()
      P[d]:div(Z)
   end

   -- Take some samples for the distributions in probability space:
   local N1 = torch.multinomial(P, N, true)
   local S1 = N1:double():sum(2):div(N):view(-1)

   -- Now convert to log space
   local logP = torch.log(P):cuda()

   local tmp = torch.zeros(D):cuda()
   local N2 = torch.zeros(D):cuda()
   for n = 1, N do
      tmp.jhu.logsample(logP:clone(), tmp)
      N2:add(tmp)
   end
   local S2 = N2:double():div(N)

   tester:eq(S1, S2, 0.1)
end

function mytest.SampleNormalized()
   local D = 5
   local N = 50000

   local P = torch.DoubleTensor(D, D):uniform(0, 1)
   for d = 1, D do
      local Z = P[d]:sum()
      P[d]:div(Z)
   end

   -- Take some samples for the distributions in probability space:
   local N1 = torch.multinomial(P, N, true)
   local S1 = N1:double():sum(2):div(N):view(-1)
   local tmp = torch.zeros(D):cuda()
   local N2 = torch.zeros(D):cuda()
   P = P:cuda()
   for n = 1, N do
      tmp.jhu.sample(P:clone(), tmp)
      N2:add(tmp)
   end
   local S2 = N2:double():div(N)

   tester:eq(S1, S2, 0.1)
end

function mytest.LogSampleUnnormalized()
   local D = 5
   local N = 50000

   local P = torch.DoubleTensor(D, D):uniform(0, 1)

   -- Take some samples for the distributions in probability space:
   local N1 = torch.multinomial(P, N, true)
   local S1 = N1:double():sum(2):div(N):view(-1)

   -- Now convert to log space
   local logP = torch.log(P):cuda()

   local tmp = torch.zeros(D):cuda()
   local N2 = torch.zeros(D):cuda()
   for n = 1, N do
      tmp.jhu.logsample(logP:clone(), tmp)
      N2:add(tmp)
   end
   local S2 = N2:double():div(N)

   tester:eq(S1, S2, 0.1)
end

function mytest.SampleUnnormalized()
   local D = 5
   local N = 50000

   local P = torch.DoubleTensor(D, D):uniform(0, 1)

   -- Take some samples for the distributions in probability space:
   local N1 = torch.multinomial(P, N, true)
   local S1 = N1:double():sum(2):div(N):view(-1)
   local tmp = torch.zeros(D):cuda()
   local N2 = torch.zeros(D):cuda()
   P = P:cuda()
   for n = 1, N do
      tmp.jhu.sample(P:clone(), tmp)
      N2:add(tmp)
   end
   local S2 = N2:double():div(N)

   tester:eq(S1,S2,0.1)
end

function mytest.EncodeDecode()
   local N = 7
   local dim = { 64, 128, 256, 512, 1024, 2048 }

   local function encode(i, j, result)
      result:copy(i)
      return result:map(j, function(s, t) return s + (t-1)*N end)
   end

   local function decode(i, o1, o2)
      for k = 1, i:size(1) do
         o1[k] = ((i[k]-1) % N) + 1
         o2[k] = math.floor(((i[k]-1) / N) + 1)
      end
   end

   for _, d in ipairs(dim) do
      local input1 = torch.LongTensor(d):random(7)
      local input2 = torch.LongTensor(d):random(7)

      local input1_gpu = input1:cuda()
      local input2_gpu = input2:cuda()

      local gold = torch.LongTensor(d)
      encode(input1, input2, gold)

      local result = torch.LongTensor(d)
      local result_gpu = result:cuda()

      -- CPU reference
      result.jhu.encode(input1, input2, result, N)

      -- GPU version
      result_gpu.jhu.encode(input1_gpu, input2_gpu, result_gpu, N)
      cutorch.synchronize()

      tester:eq(gold, result)
      tester:eq(gold, result_gpu:long())

      -- decode to get inputs back
      local decoded1 = torch.LongTensor(d)
      local decoded2 = torch.LongTensor(d)

      local decoded1_gpu = decoded1:cuda()
      local decoded2_gpu = decoded2:cuda()

      decode(result, decoded1, decoded2)

      tester:eq(decoded1, input1)
      tester:eq(decoded2, input2)

      -- CPU version
      result.jhu.decode(result, decoded1, decoded2, N)

      -- GPU version
      result_gpu.jhu.decode(result_gpu, decoded1_gpu, decoded2_gpu, N)
      cutorch.synchronize()

      tester:eq(decoded1, input1)
      tester:eq(decoded2, input2)
      tester:eq(decoded1_gpu:long(), input1)
      tester:eq(decoded2_gpu:long(), input2)
   end
end

function mytest.LogScale()
   local D = 10
   -- Vector
   local P = torch.DoubleTensor(D):uniform(0, 1):cuda()
   local logP = torch.log(P)
   logP.jhu.logscale(logP)
   local diff = math.abs(logP:sum()-1.0)
   tester:assert(diff < 1e-3, 'bad log sum: err='..diff)

   -- Matrix
   local P = torch.DoubleTensor(D, D):uniform(0, 1):cuda()
   local logP = torch.log(P)
   logP.jhu.logscale(logP)
   local diff = math.abs(logP:sum()-D)
   tester:assert(diff < 1e-3, 'bad log sum: err='..diff)
end

function mytest.Scale()
   local D = 10
   -- Vector
   local P = torch.DoubleTensor(D):uniform(0, 1):cuda()
   P.jhu.scale(P)
   local diff = math.abs(P:sum()-1.0)
   tester:assert(diff < 1e-3, 'bad log sum: err='..diff)

   -- Matrix
   local P = torch.DoubleTensor(D, D):uniform(0, 1):cuda()
   P.jhu.scale(P)
   local diff = math.abs(P:sum()-D)
   tester:assert(diff < 1e-3, 'bad log sum: err='..diff)
end

tester:add(mytest):run()
