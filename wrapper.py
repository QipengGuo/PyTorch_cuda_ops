import torch as th
import time
import numpy as np
from bmv import *
from torch.autograd import Function

B,N,H = 5000,10,64
AA = th.zeros(B, N, H).cuda()
TAA = th.zeros(B, H, N).cuda()
BB = th.zeros(B, H).cuda()
class BatchedMV(Function):
    @staticmethod
    def forward(ctx, A, B):
        A = A.contiguous()
        B = B.contiguous()
        y = bmv_forward(A, B)
        ctx.save_for_backward(A, B)
        return y

    @staticmethod
    def backward(ctx, dy):
        A, B = ctx.saved_tensors
        dA = th.einsum('bi,bj->bij', dy, B)
        #dA = th.zeros(dy.size(0), dy.size(1), B.size(1)).cuda()
        #dA = AA
        dB = bmv_forward(A.transpose(-2,-1).contiguous(), dy)
        #dB = bmv_forward(TAA, dy)
        #dB = BB
        return dA, dB

x = th.rand(B, N, H, requires_grad=True, device='cuda:0')
y = th.rand(B, H, requires_grad=True, device='cuda:0')


f1, f2, b1, b2 = [], [], [], []
for t in range(100):
    c1 = x @ y[:,:,None]
    th.cuda.synchronize()
    mt = time.time()
    c1 = x @ y[:,:,None]
    th.cuda.synchronize()
    f1.append(time.time()-mt)
    th.cuda.synchronize()
    mt = time.time()
    c2 = BatchedMV.apply(x, y)
    th.cuda.synchronize()
    f2.append(time.time()-mt)

    #print(c1.view(-1),c2.view(-1))
    assert th.allclose(c1.view(-1), c2.view(-1))

    grad = th.rand(B, N, device='cuda:0')
    th.cuda.synchronize()
    mt = time.time()
    c1.backward(grad[:,:,None])
    th.cuda.synchronize()
    b1.append(time.time()-mt)
    x_grad_clone = x.grad.clone()
    y_grad_clone = y.grad.clone()
    x.grad.zero_()
    y.grad.zero_()
    th.cuda.synchronize()
    mt = time.time()
    c2.backward(grad)
    th.cuda.synchronize()
    b2.append(time.time()-mt)
    #assert th.allclose(x.grad.view(-1), x_grad_clone.view(-1)) and th.allclose(y.grad.view(-1), y_grad_clone.view(-1))
    x.grad.zero_()
    y.grad.zero_()


print('f1',np.mean(f1), np.std(f1))
print('f2',np.mean(f2), np.std(f2))
print('b1',np.mean(b1), np.std(b1))
print('b2',np.mean(b2), np.std(b2))

