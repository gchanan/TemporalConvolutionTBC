#define TemporalConvolutionTBC_CONCAT_EXPAND(x,y,z) x ## y ## _ ## z
#define TemporalConvolutionTBC_CONCAT(x,y,z) TemporalConvolutionTBC_CONCAT_EXPAND(x,y,z)
#define TemporalConvolutionTBC_(NAME) TemporalConvolutionTBC_CONCAT(TemporalConvolutionTBC_, NAME, CReal)

#include "TemporalConvolutionTBC.cu"
//#include "THCGenerateFloatType.h"

//#include "TemporalConvolutionTBCHost.cpp"
//#include "THCGenerateDoubleType.h"
