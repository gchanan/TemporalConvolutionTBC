local ffi = require 'ffi'

ffi.cdef[[
  int TemporalConvolutionTBC_updateOutput_Float(THFloatTensor *output,THFloatTensor *input,THFloatTensor *weight,THFloatTensor *bias);
  int TemporalConvolutionTBC_updateGradInput_Float(THFloatTensor *dOutput, THFloatTensor *dInput, THFloatTensor *weight);
  int TemporalConvolutionTBC_accGradParameters_Float(THFloatTensor *input, THFloatTensor *dOutput, THFloatTensor *dWeight, THFloatTensor *dBias, double scale);
  int TemporalConvolutionTBC_updateOutput_Double(THDoubleTensor *output,THDoubleTensor *input,THDoubleTensor *weight,THDoubleTensor *bias);
  int TemporalConvolutionTBC_updateGradInput_Double(THDoubleTensor *dOutput, THDoubleTensor *dInput, THDoubleTensor *weight);
  int TemporalConvolutionTBC_accGradParameters_Double(THDoubleTensor *input, THDoubleTensor *dOutput, THDoubleTensor *dWeight, THDoubleTensor *dBias, double scale);
  int torchtbc_has_cuda();
]]

local cudaAvailable, _ = pcall(require, 'cutorch')
if cudaAvailable then
  ffi.cdef[[
    int TemporalConvolutionTBC_updateOutput_Cuda(THCState *state, THCudaTensor *output,  THCudaTensor *input,THCudaTensor *weight, THCudaTensor *bias);
    int TemporalConvolutionTBC_updateGradInput_Cuda(THCState *state, THCudaTensor *dOutput, THCudaTensor *dInput, THCudaTensor *weight);
    int TemporalConvolutionTBC_accGradParameters_Cuda(THCState *state, THCudaTensor *input, THCudaTensor *dOutput, THCudaTensor *dWeight, THCudaTensor *dBias, float scale);
    int TemporalConvolutionTBC_updateOutput_CudaDouble(THCState *state, THCudaDoubleTensor *output, THCudaDoubleTensor *input, THCudaDoubleTensor *weight, THCudaDoubleTensor *bias);
    int TemporalConvolutionTBC_updateGradInput_CudaDouble(THCState *state, THCudaDoubleTensor *dOutput, THCudaDoubleTensor *dInput, THCudaDoubleTensor *weight);
    int TemporalConvolutionTBC_accGradParameters_CudaDouble(THCState *state, THCudaDoubleTensor *input, THCudaDoubleTensor *dOutput, THCudaDoubleTensor *dWeight, THCudaDoubleTensor *dBias, double scale);
  ]]
  if cutorch.hasHalf then
    ffi.cdef[[
      int TemporalConvolutionTBC_updateOutput_CudaHalf(THCState *state, THCudaHalfTensor *output, THCudaHalfTensor *input, THCudaHalfTensor *weight, THCudaHalfTensor *bias);
      int TemporalConvolutionTBC_updateGradInput_CudaHalf(THCState *state, THCudaHalfTensor *dOutput, THCudaHalfTensor *dInput, THCudaHalfTensor *weight);
      int TemporalConvolutionTBC_accGradParameters_CudaHalf(THCState *state, THCudaHalfTensor *input, THCudaHalfTensor *dOutput, THCudaHalfTensor *dWeight, THCudaHalfTensor *dBias, float scale);
    ]]
  end
end

return ffi.load(package.searchpath('libTHTBC', package.cpath))
