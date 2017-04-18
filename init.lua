require 'torch'
require 'nn'

local tbc = tbc or {}

tbc.C = require('tbc.ffi') -- keep the loaded C object around, so that it doesn't get garbage collected

function tbc.bind(lib, base_names, type_name, state_getter)
   local ftable = {}
   local prefix = 'TemporalConvolutionTBC_'
   for i,n in ipairs(base_names) do
      local ok,v = pcall(function() return lib[prefix .. n .. '_' .. type_name] end)
      if ok then
         if state_getter then
            ftable[n] = function(...) v(state_getter, ...) end
         else
            ftable[n] = function(...) v(...) end
         end
      else
         print('not found: ' .. prefix .. n .. '_' .. type_name)
      end
   end
   return ftable
end

local function_names = {"updateOutput", "updateGradInput", "accGradParameters"}

tbc.kernels = {}
tbc.kernels['torch.FloatTensor'] = tbc.bind(tbc.C, function_names, 'Float')
tbc.kernels['torch.DoubleTensor'] = tbc.bind(tbc.C, function_names, 'Double')

torch.getmetatable('torch.FloatTensor').TBC = tbc.kernels['torch.FloatTensor']
torch.getmetatable('torch.DoubleTensor').TBC = tbc.kernels['torch.DoubleTensor']

if tbc.C.torchtbc_has_cuda() == 1 then
  tbc.kernels['torch.CudaTensor'] = tbc.bind(tbc.C, function_names, 'Cuda', cutorch.getState())
  torch.getmetatable('torch.CudaTensor').TBC = tbc.kernels['torch.CudaTensor']
end

require('tbc.TemporalConvolutionTBC')

return tbc
