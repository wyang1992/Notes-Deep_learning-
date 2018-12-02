# Pytorch_gradient

from torch import FloatTensor
from torch.autograd import Variable

a = Variable(FloatTensor([4]))
a

weights = [Variable(FloatTensor([i]), requires_grad=True) for i in (2, 5, 9, 7)]
weights

w1, w2, w3, w4 = weights

b = w1 * a
c = w2 * a
d = w3 * b + w4 * c
L = (10 - d)

L.backward()


for index, weight in enumerate(weights, start=1):
    gradient, *_ = weight.grad.data
    print(gradient)
