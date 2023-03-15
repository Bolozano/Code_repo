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
