-- luacheck: globals torch jhu nn

require('torch')
require('nn')
require('jhnn')
require('jhcunn')

local mytest = torch.TestSuite()
local mytester = torch.Tester()

local precision = 1e-5

function mytest.weightedGradUpdate()
   local weights = torch.CudaTensor({0.1,2,1,0.5})
   --print('weights:')
   --print(weights)
   local idim = 3
   local odim = 2
   local input = torch.CudaTensor(weights:size(1), idim):uniform(0,1)

   --print('\ninput:')
   --print(input)

   local module = nn.Linear(idim, odim)
   --module.weight:uniform(-0.1,0.1)
   --module.bias:uniform(-0.1,0.1)

   local refGradWeight = module.gradWeight:clone():zero()
   local refGradBias = module.gradBias:clone():zero()

   for b = 1, weights:size(1) do
      --print(input[b])
      module:zeroGradParameters()
      --print('weight = ' .. weights[b])
      module:forward(input[b])
      local dout = module.output.new():resizeAs(module.output)
      dout:fill(1)
      local din = module:backward(input[b], dout, weights[b])
      --print('gradWeight:')
      --print(module.gradWeight)
      refGradWeight:add(module.gradWeight)
      refGradBias:add(module.gradBias)
   end

   local module = jhnn.Linear(idim, odim)
   module:zeroGradParameters()
   module:forward(input)
   local dout = module.output.new():resizeAs(module.output)
   dout:fill(1)
   module:backward(input, dout, weights)

   mytester:eq(module.gradWeight, refGradWeight, 0.001)
   mytester:eq(module.gradBias, refGradBias, 0.001)
end

function mytest.weightedGradUpdateContainer()
   local weights = torch.CudaTensor({0.1,2,1,0.5})
   local idim = 3
   local odim = 2
   local fdim = 5
   local input = torch.CudaTensor(weights:size(1), idim):uniform(0,1)
   local inputClone = input:clone()

   local mlp1 = nn.Sequential()
   mlp1:add( jhnn.Linear(idim, odim) )
   mlp1:add( jhnn.Linear(odim, fdim) )
   local param1, refGrad = mlp1:getParameters()
   param1:uniform(-0.1,0.1)
   local paramClone = param1:clone()
   local refGradAcc = refGrad:clone():zero()

   for b = 1, weights:size(1) do
      refGrad:zero()
      mlp1:clearState()
      mlp1:forward(input[b])
      local dout = mlp1.output.new():resizeAs(mlp1.output)
      dout:fill(1)
      local din = mlp1:backward(input[b], dout, weights[b])
      --print('refGrad norm:' .. refGrad:norm())
      refGradAcc:add(refGrad)
   end

   local mlp2 = nn.Sequential()
   mlp2:add( jhnn.Linear(idim, odim) )
   mlp2:add( jhnn.Linear(odim, fdim) )
   local param2, grad = mlp2:getParameters()
   param2:copy(paramClone)
   grad:zero()

   mlp2:forward(inputClone)
   local dout = mlp2.output.new():resizeAs(mlp2.output)
   dout:fill(1)
   local din = mlp2:backward(inputClone, dout, weights)

   mytester:eq(grad, refGradAcc, 0.001)
end

function mytest.Jacobian()
   local ini = math.random(3,5)
   local inj_vals = {math.random(3,5), 1}  -- Also test the inj = 1 spatial case
   local input = torch.Tensor(ini):zero()
   local jac = nn.Jacobian

   for ind, inj in pairs(inj_vals) do
      local module = jhnn.Linear(ini, inj)

      -- 1D
      local err = jac.testJacobian(module,input)
      mytester:assertlt(err, precision, 'error on state ')

      local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight)
      mytester:assertlt(err, precision, 'error on weight ')

      local err = jac.testJacobianParameters(module, input, module.bias, module.gradBias)
      mytester:assertlt(err, precision, 'error on bias ')

      local err = jac.testJacobianUpdateParameters(module, input, module.weight)
      mytester:assertlt(err, precision, 'error on weight [direct update] ')

      local err = jac.testJacobianUpdateParameters(module, input, module.bias)
      mytester:assertlt(err, precision, 'error on bias [direct update] ')

      -- 2D
      local nframe = math.random(50,70)
      local input = torch.Tensor(nframe, ini):zero()

      local err = jac.testJacobian(module,input)
      mytester:assertlt(err,precision, 'error on state ')

      local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight)
      mytester:assertlt(err,precision, 'error on weight ')

      local err = jac.testJacobianParameters(module, input, module.bias, module.gradBias)
      mytester:assertlt(err,precision, 'error on weight ')

      local err = jac.testJacobianUpdateParameters(module, input, module.weight)
      mytester:assertlt(err,precision, 'error on weight [direct update] ')

      local err = jac.testJacobianUpdateParameters(module, input, module.bias)
      mytester:assertlt(err,precision, 'error on bias [direct update] ')
   end
end

-- randomize stuff
local seed = seed or os.time()
math.randomseed(seed)
torch.manualSeed(seed)
mytester:add(mytest):run()
