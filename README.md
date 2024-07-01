# torch-roller
* Roll 2d tensor to 3d and vice versa.

```python
>>> import numpy as np
>>> import pandas as pd
>>> import torch_roller as tr
>>> df = pd.DataFrame()
>>> df["A"] = [1, 2, 3, 4, 5]
>>> df["B"] = [10, 20, 30, 40, 50]
>>> df.values
array([[ 1, 10],
       [ 2, 20],
       [ 3, 30],
       [ 4, 40],
       [ 5, 50]], dtype=int64)
>>> tensor_3d = tr.roll(df, length=3)
>>> tensor_3d
tensor([[[ 1., 10.],       
         [ 2., 20.],       
         [ 3., 30.]],      

        [[ 2., 20.],       
         [ 3., 30.],       
         [ 4., 40.]],      

        [[ 3., 30.],       
         [ 4., 40.],       
         [ 5., 50.]]])     
>>> df_2d = tr.unroll(tensor_3d)
>>> df_2d.values
array([[ 1., 10.],
       [ 2., 20.],
       [ 3., 30.],
       [ 4., 40.],
       [ 5., 50.]], dtype=float32)
>>> assert np.array_equal(df.values, df_2d.values)
```
