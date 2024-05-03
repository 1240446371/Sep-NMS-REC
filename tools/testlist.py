import numpy as np
import torch
"""
testlist = [[1,2],[3,4],[5,6]]
testarray = np.asarray(testlist)  # 3*2 shape
print("testshape%s"%(testarray[2][1]))

import torch
sent_len =torch.zeros((32,1))
S=10
N=32
sent_mask = (torch.arange(S) + 1).unsqueeze(dim=0).expand(N, -1) > sent_len.unsqueeze(dim=1)
print("sent_mask%s"%sent_mask)
"""

a = torch.randn(2,2)

print("a: ",a)
c = a>0.1
b=a[c]
print("c: ",c)

print("b: ",b)

a=3
b=a**(2)
print(b)