import torch as th
import time
import numpy as np
from bmv import *
from torch.autograd import Function

NN = 30
B,N,H = 640*10*NN,5,64
AA = th.zeros(B, N, H).cuda()
TAA = th.zeros(B, H, N).cuda()
BB = th.zeros(B, H, 1).cuda()

class BatchedMV(Function):
    @staticmethod
    def forward(ctx, A, B):
        A = A.contiguous()
        B = B.contiguous()
        with th.no_grad():
            #y = th.bmm(A,B)
            y = th.bmm(A,B)
        #y = bmv_forward(A, B)
        #y = th.bmm(A,B[:,:,None])[:,:,0]
        ctx.save_for_backward(A, B)
        return y

    @staticmethod
    def backward(ctx, dy):
        A, B = ctx.saved_tensors
        with th.no_grad():
#            dA = th.bmm(dy, B.transpose(-2,-1).contiguous())
            dB = bmv_forward(A.transpose(-2,-1).contiguous(), dy[:,:,0])[:,:,None]
#            dB = th.bmm(A.transpose(-2,-1).contiguous(), dy)
#        dA = th.einsum('bi,bj->bij', dy[:,:,0], B[:,:,0])
        dA = dy * B.view(B.size(0), 1, B.size(1))
        #dA = th.zeros(dy.size(0), dy.size(1), B.size(1)).cuda()
#        dA = AA
        #dB = bmv_forward(A.transpose(-2,-1).contiguous(), dy)
#        dB = (A * dy).sum(1)[:,:,None]
        #dB = bmv_forward(TAA, dy)
#        dB = BB
        return dA, dB

x = th.rand(B, N, H, requires_grad=True, device='cuda:0')
y = th.rand(B, H, requires_grad=True, device='cuda:0')
z = th.rand(B//NN, NN, H, requires_grad=True, device='cuda:0')
zz = th.rand(B//NN, H, NN, requires_grad=True, device='cuda:0')

f1, f2, f3, f4, b1, b2, b3, b4 = [], [], [], [], [], [], [], []
for t in range(100):
    c1 = x @ y[:,:,None]
    th.cuda.synchronize()
    mt = time.time()
    #c1 = x @ y[:,:,None]
    c1 = (x * y[:,None,:]).sum(2)
    th.cuda.synchronize()
    f1.append(time.time()-mt)
    th.cuda.synchronize()
    mt = time.time()
    #c2 = BatchedMV.apply(x, y)
    c2 = th.bmm(x,y[:,:,None])
    #c2 = BatchedMV.apply(z,zz)
    th.cuda.synchronize()
    f2.append(time.time()-mt)
    th.cuda.synchronize()
    mt = time.time()
    c3 = BatchedMV.apply(x, y[:,:,None])
    th.cuda.synchronize()
    f3.append(time.time()-mt)
    mt = time.time()
    c4 =th.bmm(z,zz)
    th.cuda.synchronize()
    f4.append(time.time()-mt)

    #print(c1.view(-1),c2.view(-1))
    #assert th.allclose(c1.view(-1), c2.view(-1))

    grad = th.rand(B, N, device='cuda:0')
    th.cuda.synchronize()
    mt = time.time()
    c1.backward(grad)
    th.cuda.synchronize()
    b1.append(time.time()-mt)
    x_grad_clone = x.grad.clone()
    y_grad_clone = y.grad.clone()
    x.grad.zero_()
    y.grad.zero_()
    th.cuda.synchronize()
    mt = time.time()
    c2.backward(grad[:,:,None])
    th.cuda.synchronize()
    b2.append(time.time()-mt)
    th.cuda.synchronize()
    mt = time.time()
    #grad1 = grad1.view(grad1.size(0), 1, grad1.size(1)).view(grad1.size(0), grad1.size(1),1)
    #grad1 = grad1.squeeze(2).unsqueeze(2)
    c3.backward(grad[:,:,None])
    th.cuda.synchronize()
    b3.append(time.time()-mt)
    #assert th.allclose(x.grad.view(-1), x_grad_clone.view(-1)) and th.allclose(y.grad.view(-1), y_grad_clone.view(-1))
    x.grad.zero_()
    y.grad.zero_()
    grad = th.rand(B//NN, NN, NN, device='cuda:0')
    mt = time.time()
    c4.backward(grad)
    th.cuda.synchronize()
    b4.append(time.time()-mt)
    x.grad.zero_()
    y.grad.zero_()



print('f1',np.mean(f1), np.std(f1))
print('f2',np.mean(f2), np.std(f2))
print('f3',np.mean(f3), np.std(f3))
print('f4',np.mean(f4), np.std(f4))
print('b1',np.mean(b1), np.std(b1))
print('b2',np.mean(b2), np.std(b2))
print('b3',np.mean(b3), np.std(b3))
print('b4',np.mean(b4), np.std(b4))

