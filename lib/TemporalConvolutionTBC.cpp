// Copyright 2004-present Facebook. All Rights Reserved.
// Author: Benjamin Graham <benjamingraham@fb.com>

// Tensor formats
// Input: ilen * batchSize * inputPlanes
// Output: olen * batchSize * outputPlanes
// Weight: kw * inputPlanes * outputPlanes

#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "TemporalConvolutionTBC.cpp"
#else

extern "C" {

int TemporalConvolutionTBC_(updateOutput)(THTensor *output,
                                          THTensor *input,
                                          THTensor *weight,
                                          THTensor *bias) {
  auto ilen = THTensor_(size)(input, 0);
  auto batchSize = THTensor_(size)(input, 1);
  auto inputPlanes = THTensor_(size)(input, 2);
  auto outputPlanes = THTensor_(size)(output, 2);
  auto olen = THTensor_(size)(output, 0);
  auto kw = THTensor_(size)(weight, 0);
  int pad = (olen - ilen + kw - 1) / 2;

  THArgCheck(
      (output->nDimension == 3) && (THTensor_(size)(output, 1) == batchSize) &&
          (olen == ilen - kw + 1 + 2 * pad),
      1,
      "output has wrong dimension %ld %ld %ld", output->nDimension, THTensor_(size)(output, 1), batchSize);
  THArgCheck( (input->nDimension == 3), 2, "input has wrong dimension");
  THArgCheck(
      (weight->nDimension == 3) && (THTensor_(size)(weight, 1) == inputPlanes) &&
          (THTensor_(size)(weight, 2) == outputPlanes),
      1,
      "weight has wrong dimension");
  THArgCheck(
      (bias->nDimension == 1) && (THTensor_(size)(bias, 0) == outputPlanes),
      1,
      "bias has wrong dimension");

  real *W = THTensor_(data)(weight);
  real *B = THTensor_(data)(bias);
  real *I = THTensor_(data)(input);
  real *O = THTensor_(data)(output);

  for (int t = 0; t < olen; t++)
    for (int b = 0; b < batchSize; b++)
      for (int c = 0; c < outputPlanes; c++)
        O[t * THTensor_(stride)(output, 0) + b * THTensor_(stride)(output, 1) + c] = B[c];
  for (int k = 0; k < kw; k++) {
    int iShift = std::max(0, k - pad);
    int oShift = std::max(0, pad - k);
    int t = std::min(ilen + pad - k, olen) - oShift;
    // Note: using gemm in column-major order mode
    // input    is l*m (row-major)
    // weight   is m*r (row-major)
    // output   is l*r (row-major)
    if (t > 0)
      THBlas_(gemm)(
        'n',
        'n',
        outputPlanes, // r
        batchSize * t, // l
        inputPlanes, // m
        1, // alpha
        W + k * THTensor_(stride)(weight, 0),
        outputPlanes,
        I + iShift * THTensor_(stride)(input, 0),
        THTensor_(stride)(input, 1),
        1, // beta
        O + oShift * THTensor_(stride)(output, 0),
        THTensor_(stride)(output, 1)
      );
  }
  return 0;
}

int TemporalConvolutionTBC_(updateGradInput)(THTensor *dOutput,
                                             THTensor *dInput,
                                             THTensor *weight) {
  auto ilen = THTensor_(size)(dInput, 0);
  auto batchSize = THTensor_(size)(dInput, 1);
  auto inputPlanes = THTensor_(size)(dInput, 2);
  auto outputPlanes = THTensor_(size)(dOutput, 2);
  auto olen = THTensor_(size)(dOutput, 0);
  auto kw = THTensor_(size)(weight, 0);
  int pad = (olen - ilen + kw - 1) / 2;

  auto W = THTensor_(data)(weight);
  auto dI = THTensor_(data)(dInput);
  auto dO = THTensor_(data)(dOutput);

  for (int k = 0; k < kw; k++) {
    int iShift = std::max(0, k - pad);
    int oShift = std::max(0, pad - k);
    int t = std::min(ilen + pad - k, olen) - oShift;
    // dOutput * T(weight) -> dInput
    // Note: using gemm in column-major order mode
    // dOutput is l*m (row-major)
    // weight  is r*m (row-major)
    // dInput  is l*r (row-major)
    if (t > 0)
      THBlas_(gemm)(
          't',
          'n',
          inputPlanes, // r
          batchSize * t, // l
          outputPlanes, // m
          1, // alpha
          W + k * THTensor_(stride)(weight, 0),
          outputPlanes,
          dO + oShift * THTensor_(stride)(dOutput, 0),
          THTensor_(stride)(dOutput, 1),
          1, // beta
          dI + iShift * THTensor_(stride)(dInput, 0),
          THTensor_(stride)(dInput, 1)
        );
  }
  return 0;
}

int TemporalConvolutionTBC_(accGradParameters)(THTensor *input,
                                               THTensor *dOutput,
                                               THTensor *dWeight,
                                               THTensor *dBias,
                                               accreal scale) {
  auto ilen = THTensor_(size)(input, 0);
  auto batchSize = THTensor_(size)(input, 1);
  auto inputPlanes = THTensor_(size)(input, 2);
  auto outputPlanes = THTensor_(size)(dOutput, 2);
  auto olen = THTensor_(size)(dOutput, 0);
  auto kw = THTensor_(size)(dWeight, 0);
  int pad = (olen - ilen + kw - 1) / 2;

  auto dW = THTensor_(data)(dWeight);
  auto dB = THTensor_(data)(dBias);
  auto I = THTensor_(data)(input);
  auto dO = THTensor_(data)(dOutput);

  for (int t = 0; t < olen; t++)
    for (int b = 0; b < batchSize; b++)
      for (int c = 0; c < outputPlanes; c++)
        dB[c] += dO[t * THTensor_(stride)(dOutput, 0) + b * THTensor_(stride)(dOutput, 1) + c];

  for (int k = 0; k < kw; k++) {
    int iShift = std::max(0, k - pad);
    int oShift = std::max(0, pad - k);
    int t = std::min(ilen + pad - k, olen) - oShift;
    // Note: using gemm in column-major order mode
    // Input    is m*l (row-major)
    // dOutput  is m*r (row-major)
    // dWeight  is l*r (row-major)
    if (t > 0)
      THBlas_(gemm)(
          'n',
          't',
          outputPlanes, // r
          inputPlanes, // l
          batchSize * t, // m
          scale, // alpha
          dO + oShift * THTensor_(stride)(dOutput, 0),
          THTensor_(stride)(dOutput, 1),
          I + iShift * THTensor_(stride)(input, 0),
          THTensor_(stride)(input, 1),
          1, // beta
          dW + k * THTensor_(stride)(dWeight, 0),
          outputPlanes
        );
  }
  return 0;
}

}

#endif
