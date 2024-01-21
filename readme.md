## Levenshtein Distance B/w Words Implementation with Python 

```python
import numpy as np
import string
```


```python
# prepare pre-processing

delete_dict = {sp_character: '' for sp_character in string.punctuation}
delete_dict[' '] = ''
table = str.maketrans(delete_dict)

def normalise(word):
    return word.lower()\
            .strip()\
            .translate(table)
```


```python
def min_edit_distance(source, target):
    """
     @params
        - source: the word which is typed
        - target: words in dictionary 
    """


    source = np.array([k for k in source])
    target = np.array([k for k in target])

    len_target = len(target)
    len_source = len(source)

    # 0 matrix with len(source) x len(target)
    sol = np.zeros((len_source, len_target), dtype=int)
    # first row 

    sol[0] = [k for k in range(len_target)]
    sol[:,0] = [k for k in range(len_source)]
    for col in range(1, len_target):
        for row in range(1, len_source):
            if target[col] != source[row]:
               sol[row, col] = min(sol[row - 1, col], sol[row, col - 1]) + 1
            else:
                sol[row, col] = sol[row - 1, col-1]
    return sol[len_source - 1, len_target - 1], sol

source = 'Danial'
target = 'Danielo'

# normalise text
source = normalise(source)
target = normalise(target)

distance, Distancematrix =  min_edit_distance(source, target)

print(f"Levenshtien Distance: {distance}\n")
print(f"Matrix: \n")
display(Distancematrix)
```

    Levenshtien Distance: 3
    
    Matrix: 
    



    array([[0, 1, 2, 3, 4, 5, 6],
           [1, 0, 1, 2, 3, 4, 5],
           [2, 1, 0, 1, 2, 3, 4],
           [3, 2, 1, 0, 1, 2, 3],
           [4, 3, 2, 1, 2, 3, 4],
           [5, 4, 3, 2, 3, 2, 3]])



```python
import pandas as pd 
df = pd.DataFrame(Distancematrix, columns=[k for k in target], index=[k for k in source])
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>d</th>
      <th>a</th>
      <th>n</th>
      <th>i</th>
      <th>e</th>
      <th>l</th>
      <th>o</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>d</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>a</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>n</th>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>i</th>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>a</th>
      <td>4</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>l</th>
      <td>5</td>
      <td>4</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
x, y = df.shape
x, y = x - 1, y - 1
```


```python
import matplotlib.pyplot as plt 
import seaborn as sns 
from matplotlib.pyplot import Rectangle
plt.figure(figsize=(18, 4))
ax = sns.heatmap(df, cmap='crest', annot=True, fmt='g')
ax.tick_params(labeltop=True)
ax.add_patch(Rectangle(
         (y, x),
         1.0,
         1,
         edgecolor='red',
         fill=False,
         lw=6
     ))
```




    <matplotlib.patches.Rectangle at 0x7f907df14fd0>




    
![png](main_files/main_5_1.png)
    



```python

```
