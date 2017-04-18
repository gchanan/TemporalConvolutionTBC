// Copyright 2016 Facebook

#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/TemporalConvolutionTBCHost.cpp"
#else

using namespace std;

namespace {

extern "C" {

int TemporalConvolutionTBC_(updateOutput)(THCState *state,
                                          THCTensor *output,
                                          THCTensor *input,
                                          THCTensor *weight,
                                          THCTensor *bias) {
  THAssert(THCTensor_(checkGPU)(state, 4, input, output, weight, bias));

  auto inputDev = toDeviceTensor<real, 3>(state, input);
  auto outputDev = toDeviceTensor<real, 3>(state, output);
  auto weightDev = toDeviceTensor<real, 3>(state, weight);
  auto biasDev = toDeviceTensor<real, 1>(state, bias);

  detail::runTemporalConvolutionTBC_updateOutput(
      state, inputDev, outputDev, weightDev, biasDev);

  return 0;
}

int TemporalConvolutionTBC_(updateGradInput)(THCState *state,
                                             THCTensor *dOutput,
                                             THCTensor *dInput,
                                             THCTensor *weight) {
  THAssert(THCTensor_(checkGPU)(state, 3, dInput, dOutput, weight));

  auto dInputDev = toDeviceTensor<real, 3>(state, dInput);
  auto dOutputDev = toDeviceTensor<real, 3>(state, dOutput);
  auto weightDev = toDeviceTensor<real, 3>(state, weight);

  detail::runTemporalConvolutionTBC_updateGradInput(
      state, dInputDev, dOutputDev, weightDev);

  return 0;
}

int TemporalConvolutionTBC_(accGradParameters)(THCState *state,
                                               THCTensor *input,
                                               THCTensor *dOutput,
                                               THCTensor *dWeight,
                                               THCTensor *dBias,
                                               accreal scale) {
  THAssert(THCTensor_(checkGPU)(state, 4, input, dOutput, dWeight, dBias));

  auto inputDev = toDeviceTensor<real, 3>(state, input);
  auto dOutputDev = toDeviceTensor<real, 3>(state, dOutput);
  auto dWeightDev = toDeviceTensor<real, 3>(state, dWeight);
  auto dBiasDev = toDeviceTensor<real, 1>(state, dBias);

  detail::runTemporalConvolutionTBC_accGradParameters(
      state, inputDev, dOutputDev, dWeightDev, dBiasDev, scale);

  return 0;
}

}
} // namespace

#endif
