# re查找

## 查找[ ]括起来的list
`re.search(r'\[.*\]',intent_string).group(0)                      
re.search(r'\[.*\]',intent_string).span(0)`

## 把一个被字母或者数字夹起来的“替换为‘（即不替换句子头尾的”）     
`pattern = re.compile("([0-9A-Za-z])\"([0-9A-Za-z])")             
re.sub(pattern,r"\1'\2",intent_string)`
## 查找替换不加莫名的\，如果不是+"\","会有莫名的\
`pattern = re.compile("([^\"]),")
intent_string1=re.sub(r"([^\"]),",r"\1" +"\",",intent_string)`

## re查找最右侧的一个字母数字下划线字符，把右边不是的删掉 
`tt[:re.search(r'[^a-zA-Z0-9_]*$',tt).span(0)[0]]      #主要是最右侧，\W=[^a-zA-Z0-9_] \w=[a-zA-Z0-9_]   `        


## 网址解析
`string = "https://example.com/path/"
result = re.findall(r'/([^/]+)', string)
print(result)`
