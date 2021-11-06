# -*- encoding:utf-8 -*-
# Author: Lg
# Date: 19/5/4
'''
    实现带约束的差分进化算法（DE），对多个分类器的融合权重进行优化
'''
import sys
import random
import numpy as np 
import matplotlib.pyplot as plt 

class Unit(object):
    def __init__(self, fit_fun, x_min, x_max, dim):
        self.pos = np.array([x_min + random.random()*(x_max - x_min) for i in range(dim)])
        self.mutation = np.array([0.0 for i in range(dim)])   # 个体变异后的向量
        self.crossover = np.array([0.0 for i in range(dim)])   # 个体交叉后的向量
        self.fitnessValue = fit_fun(self.pos)   # 个体适应度

    def setPos(self, i, value):
        self.pos[i] = value

    def getPos(self):
        return self.pos

    def setMutation(self, i, value):
        self.mutation[i] = value

    def getMutation(self):
        return self.mutation

    def setCrossover(self, i, value):
        self.crossover[i] = value

    def getCrossover(self):
        return self.crossover

    def setFitnessValue(self, value):
        self.finessValue = value

    def getFitnessValue(self):
        return self.fitnessValue



class DE(object):
    def __init__(self, fit_fun, dim, size, iter_num, x_min, x_max, best_fitness_value=float('inf'), F=0.5, CR=0.8):
        self.F = F
        self.CR = CR
        self.fit_fun = fit_fun
        self.dim = dim   # 维度
        self.size = size   # 个体总数
        self.iter_num = iter_num   # 迭代次数
        self.x_min = x_min
        self.x_max = x_max
        self.best_fitness_value = best_fitness_value
        self.best_position = [0.0 for i in range(self.dim)]   # 全局最优解
        self.fitness_val_list = []   # 每次迭代最优适应值

        self.unit_list = [Unit(self.fit_fun, self.x_min, self.x_max, self.dim) for i in range(self.size)]   # 初始化种群


    #　变异
    def doMutation(self):
        for i in range(self.size):
            r1 = r2 = r3 = 0
            while r1==i or r2==i or r3==i or r2==r1 or r3==r1 or r3==r2:
                r1 = random.randint(0, self.size - 1)
                r2 = random.randint(0, self.size - 1)
                r3 = random.randint(0, self.size - 1)
            mutation = self.unit_list[r1].getPos() + self.F * (self.unit_list[r2].getPos() - self.unit_list[r3].getPos())
            for j in range(self.dim):
                # if (self.x_min < mutation[j] < self.x_max) and (np.sum(mutation) == 1):    # 判断变异后的值是否满足边界条件
                if (self.x_min < mutation[j] < self.x_max):
                    self.unit_list[i].setMutation(j, mutation[j])
                else:
                    rand_value = self.x_min + random.random()*(self.x_max - self.x_min)
                    self.unit_list[i].setMutation(j, rand_value)

    # 交叉
    def doCrossover(self):
        for u in self.unit_list:
            for d in range(self.dim):
                rand_d = random.randint(0, self.dim - 1)
                rand_float = random.random()
                if rand_float <= self.CR or rand_d==d:
                    u.setCrossover(d, u.getMutation()[d])
                else:
                    u.setCrossover(d, u.getPos()[d])

    # 选择
    def doSelection(self):
        for u in self.unit_list:
            new_fitness_value = self.fit_fun(u.getCrossover())
            # print('new_fitness_value: {}'.format(new_fitness_value.shape))
            if new_fitness_value < u.getFitnessValue():
                u.setFitnessValue(new_fitness_value)
                for i in range(self.dim):
                    u.setPos(i, u.getCrossover()[i])
            # print(self.best_fitness_value)
            if new_fitness_value < self.best_fitness_value:
                self.best_fitness_value = new_fitness_value
                for d in range(self.dim):
                    self.best_position[d] = u.getCrossover()[d]

    def doUpdate(self):
        for i in range(self.iter_num):
            print('\n----------- The number of iteration is: {} -----------\n'.format(i + 1))
            self.doMutation()
            self.doCrossover()
            self.doSelection()
            self.fitness_val_list.append(self.best_fitness_value)
        return self.fitness_val_list, self.best_position



def fit_fun(X):  # 适应函数
    return -np.abs(np.sin(X[0]) * np.cos(X[1]) * np.exp(np.abs(1 - np.sqrt(X[0] ** 2 + X[1] ** 2) / np.pi)))
    #return X[0]**2 + X[1]**2 + X[2]**2

if __name__ == '__main__':
    dim = 3
    size = 10
    iter_num = 500
    x_max = 10
    x_min = 0
    max_vel = 0.05
    de = DE(fit_fun, dim, size, iter_num, -x_max, x_max)
    fit_var_list2, best_pos2 = de.doUpdate()
    print('DE best solution:' + str(best_pos2))
    print("DE:" + str(fit_var_list2[-1]))
    plt.plot(np.linspace(0, iter_num, iter_num), fit_var_list2, c="G", alpha=0.5, label="DE")
    

    plt.legend()  
    plt.show()
    
