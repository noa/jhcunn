-- luacheck: globals torch jhu nn

require('torch')
require('nn')
require('jhcunn')

local mytest = torch.TestSuite()
local mytester = torch.Tester()

local precision = 1e-5

function mytest.weightedGradUpdate()
   local weights = torch.CudaTensor({0.1,2,1,0.5})
   local idim = 5
   local odim = 3
   local batchSize = 4
   local batchLen = 1
   local input = torch.LongTensor(batchSize, batchLen):random(1, idim)
   local module = nn.LookupTable(idim, odim)
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
      refGradWeight:add(module.gradWeight)
   end

   --print('input ndim = ' .. input:nDimension())

   local module = jhnn.LookupTable(idim, odim)
   module:zeroGradParameters()

   input = input:cuda()

   module:forward(input)
   local dout = module.output.new():resizeAs(module.output)
   dout:fill(1)

   dout = dout:cuda()

   module:backward(input, dout, weights)

   mytester:eq(module.gradWeight, refGradWeight, 0.001)
end

function mytest.LookupTable()
   local totalIndex = math.random(6,9)
   local nIndex = math.random(3,5)
   local entry_size = math.random(2,5)
   local jac = nn.Jacobian

   local function dotest(module, input, minval, maxval)
       local output = module:forward(input)
       module:backwardUpdate(input, output, 0.1)
       input:zero()

       -- 1D
       local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight, minval, maxval)
       mytester:assertlt(err,precision, '1D error on weight ')

       local err = jac.testJacobianUpdateParameters(module, input, module.weight, minval, maxval)
       mytester:assertlt(err,precision, '1D error on weight [direct update] ')

       module.gradWeight:zero()
       for t,err in pairs(jac.testAllUpdate(module, input, 'weight', 'gradWeight')) do
          mytester:assertlt(err, precision, string.format(
                             '1D error on weight [%s]', t))
       end

       -- 2D
       local nframe = math.random(2,5)
       local input = torch.IntTensor(nframe, nIndex):zero()

       local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight, minval, maxval)
       mytester:assertlt(err,precision, '2D error on weight ')

       local err = jac.testJacobianUpdateParameters(module, input, module.weight, minval, maxval)
       mytester:assertlt(err,precision, '2D error on weight [direct update] ')

       module.gradWeight:zero()
       for t,err in pairs(jac.testAllUpdate(module, input, 'weight', 'gradWeight')) do
          mytester:assertlt(err, precision, string.format(
                             '2D error on weight [%s]', t))
       end

       -- IO
       module.gradInput = torch.Tensor(3,4):zero() --fixes an error
       local ferr,berr = jac.testIO(module,input,minval,maxval)
       mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
       mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')

       -- accUpdate
       module:accUpdateOnly()
       mytester:assert(not module.gradWeight, 'gradWeight is nil')
       module:float()
       local output = module:forward(input)
       module:backwardUpdate(input, output, 0.1)
   end
   -- test without padding
   local input = torch.randperm(totalIndex):narrow(1,1,nIndex):int()
   local module = jhnn.LookupTable(totalIndex, entry_size)
   dotest(module, input, 1, totalIndex)
   -- test with padding set to 1, but no padding in inputs
   local input = torch.randperm(totalIndex):narrow(1,1,nIndex):int()
   local module = jhnn.LookupTable(totalIndex, entry_size, 1)
   dotest(module, input, 2, totalIndex)
   -- test whether padding weights remain unchanged
   local paddingValue = math.random(totalIndex)
   local module = jhnn.LookupTable(totalIndex, entry_size, paddingValue)
   local padw = module.weight:select(1,paddingValue):fill(1)
   local padw_sum = padw:sum()
   local input = torch.IntTensor(nIndex)
   for i = 1, 100 do
       input:apply(
       function() -- set randomly half of the input as padding
           if torch.random(2) == 1 then return paddingValue end
           return torch.random(totalIndex)
       end)
       local y = module:updateOutput(input)
       module:updateGradInput(input, y)
       module:accUpdateGradParameters(input, y, 0.1)
   end
   local err = padw_sum - padw:sum()
   mytester:assertlt(err,precision, 'padding update error ')
end

-- randomize stuff
local seed = seed or os.time()
math.randomseed(seed)
torch.manualSeed(seed)
mytester:add(mytest):run()
