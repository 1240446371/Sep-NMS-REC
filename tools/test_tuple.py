import numpy as np
import torch
list1 = []
tuple1 = ([1,2,3,4],0.7)
tuple2 = ([5,6,7,8],0.6)
list1.append(tuple1)
list1.append(tuple2)
for box,score in list1:
   print("box %s score %s"%(box,score))
   
   
list1 = [[1,2,3,4],[5,6,7,8]]
array1 = np.asarray(list1)
#score = [0.1,0.2]
array1 = torch.from_numpy(array1)
list2 = [1,2,3,4]
array2 = np.asarray(list2)

print("array shape%s"%(array1.shape,))
print("array t()%s"%len(array1.t()))
print("array shape t()%s"%(array1.shape,))


"""print("len array %s"%len(array))
for i in  range(0,len(array)):
    for j in range(len(array[0])):
        print("i %s, j %s array %s"%(i,j,array[i][j]))"""


"""for i in range(0,len(lista)):
    print("list%s,score%s"%(lista[i],score[i]))"""