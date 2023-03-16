torch矩阵维度转置

```
x = torch.randn(2, 3, 5)
x.size()
torch.permute(x, (2, 0, 1)).size()
```

torch乘法

点乘 element wise  

```
*     #a:shape(n,m)   b(l,n,m)
```

