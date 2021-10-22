import time

dict = {}
array = [i for i in range(100000)]
s = time.time()
for i in range(100000):
    dict[i] = i
print(time.time()-s)

s = time.time()
for i in range(100000):
    a = dict[i]
print(time.time()-s)


s = time.time()
for i in range(100000):
    a = array[i]
print(time.time()-s)