// Copyright 2016 Facebook

#include "THC.h"
#include "THCTensor.h"
#include "THCDeviceTensor.cuh"
#include "THCDeviceTensorUtils.cuh"
#include "THCDeviceUtils.cuh"
#include "TemporalConvolutionTBC.cuh"

#include <cuda_runtime.h>


using namespace std;

namespace {

extern "C" {

int TemporalConvolutionTBC_updateOutput_Cuda(THCState *state,
                                             THCudaTensor *output,
                                             THCudaTensor *input,
                                             THCudaTensor *weight,
                                             THCudaTensor *bias) {
  THAssert(THCudaTensor_checkGPU(state, 4, input, output, weight, bias));

  auto inputDev = toDeviceTensor<float, 3>(state, input);
  auto outputDev = toDeviceTensor<float, 3>(state, output);
  auto weightDev = toDeviceTensor<float, 3>(state, weight);
  auto biasDev = toDeviceTensor<float, 1>(state, bias);

  detail::runTemporalConvolutionTBC_updateOutput(
      state, inputDev, outputDev, weightDev, biasDev);

  return 0;
}

int TemporalConvolutionTBC_updateGradInput_Cuda(THCState *state,
                                                THCudaTensor *dOutput,
                                                THCudaTensor *dInput,
                                                THCudaTensor *weight) {
  THAssert(THCudaTensor_checkGPU(state, 3, dInput, dOutput, weight));

  auto dInputDev = toDeviceTensor<float, 3>(state, dInput);
  auto dOutputDev = toDeviceTensor<float, 3>(state, dOutput);
  auto weightDev = toDeviceTensor<float, 3>(state, weight);

  detail::runTemporalConvolutionTBC_updateGradInput(
      state, dInputDev, dOutputDev, weightDev);

  return 0;
}

int TemporalConvolutionTBC_accGradParameters_Cuda(THCState *state,
                                                  THCudaTensor *input,
                                                  THCudaTensor *dOutput,
                                                  THCudaTensor *dWeight,
                                                  THCudaTensor *dBias,
                                                  float scale) {
  THAssert(THCudaTensor_checkGPU(state, 4, input, dOutput, dWeight, dBias));

  auto inputDev = toDeviceTensor<float, 3>(state, input);
  auto dOutputDev = toDeviceTensor<float, 3>(state, dOutput);
  auto dWeightDev = toDeviceTensor<float, 3>(state, dWeight);
  auto dBiasDev = toDeviceTensor<float, 1>(state, dBias);

  detail::runTemporalConvolutionTBC_accGradParameters(
      state, inputDev, dOutputDev, dWeightDev, dBiasDev, scale);

  return 0;
}

}
} // namespace
