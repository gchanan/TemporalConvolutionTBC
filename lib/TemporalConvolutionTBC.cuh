// Copyright 2004-present Facebook. All Rights Reserved.

#include "THCGeneral.h"
#include "THCDeviceTensor.cuh"

#include <algorithm>

namespace detail {

void runTemporalConvolutionTBC_updateOutput(
    THCState* state,
    const THCDeviceTensor<float, 3>& input,
    const THCDeviceTensor<float, 3>& output,
    const THCDeviceTensor<float, 3>& weight,
    const THCDeviceTensor<float, 1>& bias);

void runTemporalConvolutionTBC_updateGradInput(
    THCState* state,
    const THCDeviceTensor<float, 3>& dInput,
    const THCDeviceTensor<float, 3>& dOutput,
    const THCDeviceTensor<float, 3>& weight);

void runTemporalConvolutionTBC_accGradParameters(
    THCState* state,
    const THCDeviceTensor<float, 3>& input,
    const THCDeviceTensor<float, 3>& dOutput,
    const THCDeviceTensor<float, 3>& dWeight,
    const THCDeviceTensor<float, 1>& dBias,
    float scale);

}
