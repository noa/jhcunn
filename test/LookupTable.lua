-- luacheck: globals torch jhu nn

require('torch')
require('nn')
require('jhcunn')

local mytest = torch.TestSuite()
local mytester = torch.Tester()

local precision = 1e-5

local precision_forward = 1e-4
local precision_backward = 1e-2

function mytest.weightedGradUpdate()
   local weights = torch.CudaTensor({0.1,2,1,0.5})
   local idim = 5
   local odim = 3
   local batchSize = 4
   local batchLen = 1
   local input = torch.LongTensor(batchSize, batchLen):random(1, idim)
   local module = nn.LookupTable(idim, odim):cuda()
   local refGradWeight = module.gradWeight:clone():zero()

   --print('input:')
   --print(input)

   for b = 1, weights:size(1) do
      module:zeroGradParameters()
      local out = module:forward(input[b])
      --print(out)
      local dout = module.output.new():resizeAs(module.output)
      dout:fill(1)
      local din = module:backward(input[b], dout, weights[b])
      cutorch.synchronize()
      refGradWeight:add(module.gradWeight)
   end

   --print('input ndim = ' .. input:nDimension())

   local module = jhnn.LookupTable(idim, odim):cuda()
   module:zeroGradParameters()

   input = input:cuda()

   module:forward(input)
   local dout = module.output.new():resizeAs(module.output)
   dout:fill(1)

   dout = dout:cuda()

   module:backward(input, dout, weights)
   cutorch.synchronize()

   mytester:eq(module.gradWeight, refGradWeight, 0.001)
end

function mytest.LookupTable_forward()
   local nVocab = 10000
   local nDim = 100
   local nInput = 1000
   local nloop = 10

   local tm = {}
   local title = string.format('LookupTable forward %d x %d', nVocab, nDim)

   local input = torch.LongTensor(nInput):random(nVocab)
   local sconv = jhnn.LookupTable(nVocab, nDim)
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
       groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real

   input = input:cuda()
   local gconv = sconv:cuda()
   local rescuda = gconv:forward(input)
   a:reset()
   for i = 1,nloop do
       rescuda = gconv:forward(input)
   end
   cutorch.synchronize()
   tm.gpu = a:time().real

   --local error = rescuda:float() - groundtruth
   --mytester:assertlt(error:abs():max(), precision_forward, 'error on state')
   mytester:eq(rescuda:float(), groundtruth:float(), precision_forward, 'error on state')
end

function mytest.LookupTable_backward()
   local grid = {
      nInput = {10, 101, 1000, 10007},
      nVocab = {100, 10000},
      nDim = {97, 255},
      scaleGradByFreq = {false, true},
      batch = {false, true},
      paddingValue = {0, 1},
   }

   for itr = 1, 10 do
      -- Randomly sample from grid of parameters
      local s = {}
      for k, v in pairs(grid) do
         s[k] = v[torch.random(#v)]
      end

      local input, gradOutput
      if s.batch then
         input = torch.LongTensor(s.nInput, 5):random(s.nVocab)
         gradOutput = torch.randn(s.nInput, 5, s.nDim)
      else
         input = torch.LongTensor(s.nInput):random(s.nVocab)
         gradOutput = torch.randn(s.nInput, s.nDim)
      end

      local sconv = jhnn.LookupTable(s.nVocab, s.nDim, s.paddingValue)
      local gconv = sconv:clone():cuda()
      if s.scaleGradByFreq then
         sconv = sconv:scaleGradByFreq()
         gconv = gconv:scaleGradByFreq()
      end

      sconv:forward(input)
      sconv:backward(input, gradOutput)

      input = input:cuda()
      gradOutput = gradOutput:cuda()
      gconv:forward(input)
      gconv:backward(input, gradOutput)

      -- print(gconv:type())
      -- print(gconv.gradWeight:size())
      -- print(sconv:type())
      -- print(sconv.gradWeight:size())

      -- local weightGradError = gconv.gradWeight:float() - sconv.gradWeight
      -- mytester:assertlt(weightGradError:abs():max(), precision_backward,
      mytester:eq(gconv.gradWeight:float(), sconv.gradWeight:float(), precision_backward,
         'error on weight for size ' .. tostring(s.nInput) ..
          ' nVocab: ' .. tostring(s.nVocab) ..
          ' nDim ' .. tostring(s.nDim) ..
          ' scaleGradByFreq: ' .. tostring(s.scaleGradByFreq) ..
          ' batch: ' .. tostring(s.batch) ..
          ' paddingValue: ' .. tostring(s.paddingValue))
   end

   local nVocab = 10000
   local nDim = 128
   local nInput = 1000
   local nloop = 100
   local tm = {}
   local title = string.format('LookupTable backward %d x %d', nVocab, nDim, nInput)

   local input = torch.LongTensor(nInput):random(nVocab)
   local gradOutput = torch.randn(nInput, nDim)
   local sconv = jhnn.LookupTable(nVocab, nDim)
   local gconv = sconv:clone():cuda()

   sconv:forward(input)
   sconv:backward(input, gradOutput)
   local a = torch.Timer()
   for i = 1,nloop do
       sconv:backward(input, gradOutput)
   end
   tm.cpu = a:time().real

   input = input:cuda()
   gradOutput = gradOutput:cuda()
   gconv:forward(input)
   gconv:backward(input, gradOutput)
   a:reset()
   for i = 1,nloop do
       gconv:backward(input, gradOutput)
   end
   cutorch.synchronize()
   tm.gpu = a:time().real

   -- local weightGradError = gconv.gradWeight:float() - sconv.gradWeight
   -- mytester:assertlt(weightGradError:abs():max(), precision_backward, 'error on weight')
   mytester:eq(gconv.gradWeight:float(), sconv.gradWeight:float(), precision_backward, 'error on weight')
end

-- randomize stuff
local seed = seed or os.time()
math.randomseed(seed)
torch.manualSeed(seed)
mytester:add(mytest):run()
