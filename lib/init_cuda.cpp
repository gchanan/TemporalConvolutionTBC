#include <THCGeneral.h>

#include <algorithm>

//#define TemporalConvolutionTBC_CONCAT_EXPAND(x,y,z) x ## y ## _ ## z
//#define TemporalConvolutionTBC_CONCAT(x,y,z) TemporalConvolutionTBC_CONCAT_EXPAND(x,y,z)
//#define TemporalConvolutionTBC_(NAME) TemporalConvolutionTBC_CONCAT(TemporalConvolutionTBC_, NAME, Real)

#include "generic/TemporalConvolutionTBCHost.cpp"
//#include "THCGenerateFloatTypes.h"
