local ffi = require 'ffi'

ffi.cdef[[
int TemporalConvolutionTBC_updateOutput_Float(THFloatTensor *output,THFloatTensor *input,THFloatTensor *weight,THFloatTensor *bias);
int TemporalConvolutionTBC_updateGradInput_Float(THFloatTensor *dOutput, THFloatTensor *dInput, THFloatTensor *weight);
int TemporalConvolutionTBC_accGradParameters_Float(THFloatTensor *input, THFloatTensor *dOutput, THFloatTensor *dWeight, THFloatTensor *dBias, double scale);
int TemporalConvolutionTBC_updateOutput_Double(THDoubleTensor *output,THDoubleTensor *input,THDoubleTensor *weight,THDoubleTensor *bias);
int TemporalConvolutionTBC_updateGradInput_Double(THDoubleTensor *dOutput, THDoubleTensor *dInput, THDoubleTensor *weight);
int TemporalConvolutionTBC_accGradParameters_Double(THDoubleTensor *input, THDoubleTensor *dOutput, THDoubleTensor *dWeight, THDoubleTensor *dBias, double scale);
]]

return ffi.load(package.searchpath('libTHTBC', package.cpath))
