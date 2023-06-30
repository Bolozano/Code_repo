# MD Pandas 是真的难用

<img width="402" alt="image" src="https://user-images.githubusercontent.com/65296071/225220939-fe765ad7-3332-4adc-be0d-1e76ff230651.png">


```
pp=list(df1.columns)
print(pp)
def f(x):
    for col in pp:
        if x[col]>0.5:
            return col
    return 
df1.apply(f,axis=1)
```




series.reindex,   dataframe.reindex    如果新索引没有对应的值，默认为nan，减少索引的话，相当于切片。

pandas中的shift函数
shift(period,freq,axis)
index不动移动数据，移动后没有值的赋值为Nan。



Dataframe.groupby().shift()要考虑到grouby后在一个groupby组内的元素已经去掉nan了，所以shift（2）会有问题。
