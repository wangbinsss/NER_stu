import re
with open('mydata/jixunying.txt','r+',encoding='utf-8') as f:
    data=f.read()


rr=re.compile(',/O',re.S)
sentences=rr.split(data)
for x in sentences:
    print(len(x))
    with open('mydata/data.txt', 'w+', encoding='utf-8') as t:
        t.write(x)

print(len(sentences))
t.close()
f.close()