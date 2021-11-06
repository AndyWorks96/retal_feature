# _*_ encoding:utf-8 _*_

import math

data_path = './123.txt'

def Distance(datas):
    temp = [d**2 for d in datas]
    temp = sum(temp)
    return math.sqrt(temp)

min_data = [99999999, 99999999, 99999999]
min_index = -1
with open(data_path,'r') as f:
    i = 0
    for line in f:
        i += 1
        temp = line.strip('\n').split(' ')
        # print(float(temp[0]))
        # print(temp.split(' '))
        datas = [float(d) for d in temp if d not in '']
        # distance = Distance(datas)
        if Distance(datas) < Distance(min_data):
            min_data = datas
            min_index = i
            

        
print(min_index)
print(min_data)
    
