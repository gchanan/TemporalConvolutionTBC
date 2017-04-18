# TemporalConvolutionTBC

## Build
```
luarocks make rocks/tbc-scm-1.rockspec
```

## Test

torch.FloatTensor:
```
th> require 'tbc'
th> nn.TemporalConvolutionTBC.test().checkforwardbackward()
delta	5.4887647940155e-07
delta	1.3836279580393e-06
delta	6.9185068266942e-07
delta	1.6403471597922e-07
```

torch.DoubleTensor:
```
th> require 'tbc'
th> nn.TemporalConvolutionTBC.test().checkforwardbackward(nil, nil, nil, nil, nil, nil, 'torch.DoubleTensor')
delta	7.9853147616588e-16
delta	2.3127072466641e-15
delta	1.1667500212561e-15
delta	4.9230179827439e-16
```

torch.CudaTensor:
```
th> require 'tbc'
th> nn.TemporalConvolutionTBC.test().checkforwardbackward(nil, nil, nil, nil, nil, nil, 'torch.CudaTensor')
delta	7.5263126531217e-07
delta	9.1754044370814e-07
delta	7.4550386618305e-07
delta	6.8602038801142e-07
```
torch.CudaDoubleTensor:
```
th> require 'tbc'
th> nn.TemporalConvolutionTBC.test().checkforwardbackward(nil, nil, nil, nil, nil, nil, 'torch.CudaDoubleTensor')
delta	1.2000670972572e-15
delta	1.9863067950491e-15
delta	2.7305562685917e-15
delta	1.1856084446228e-15
```
