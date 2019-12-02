import numpy as np

theta=np.zeros(5)
print(theta.T.shape)

a=np.arange(2)
b=np.arange(2)
print(a)
print(b)
print(a*b)
print(a.dot(b.T))

mx_a=np.arange(4).reshape(2,2)

print(mx_a.dot(a))
