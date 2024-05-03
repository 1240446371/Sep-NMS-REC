
import pandas as pd
import numpy as np

 
scores = [1, 0.5, 0.2, 0.01, 0.02, 0.1, 0.05, 0.0001, 0.0002, 0.0015, 0.015, 0.00001,0.5,0.5]
#scores = np.array(scores)

#bins = [0.0, 0.01, 0.2, 0.4, 0.6, 0.8,1.0]
label1=['0', '1', '2','3','4']
#scores = [1, 0.5, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2] 
cats = pd.qcut(scores, 5 ,duplicates='drop',labels= label1)
print(cats.codes)
#print("cats%s"%type(cats.codes))
#print("cats1%s"%cats[1])


