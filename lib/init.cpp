#include <THGeneral.h>
#include <TH.h>

#include <algorithm>

#define TemporalConvolutionTBC_CONCAT_EXPAND(x,y,z) x ## y ## _ ## z
#define TemporalConvolutionTBC_CONCAT(x,y,z) TemporalConvolutionTBC_CONCAT_EXPAND(x,y,z)
#define TemporalConvolutionTBC_(NAME) TemporalConvolutionTBC_CONCAT(TemporalConvolutionTBC_, NAME, Real)

extern "C" {
int torchtbc_has_cuda() {
#ifdef TORCH_TBC_CUDA
    return 1;
#else
    return 0;
#endif
}
}

#include "generic/TemporalConvolutionTBC.cpp"
#include "THGenerateFloatTypes.h"
