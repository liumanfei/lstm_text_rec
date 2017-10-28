#coding = gbk
import os
import numpy as np

def read_dictionary(dic_dir='../data/dic_7356.txt'):
	dictionary={}
	with open(dic_dir,'r')as f:
		# f.encode('utf-8').strip()
		lines = f.read()
		print(lines)
		input('wait')
		# for id,line in lines:
		# 	dictionary[line] = id 
		# print(dictionary[0:10])
	return dictionary

from numpy import *  
import matplotlib  
import matplotlib.pyplot as plt 
import random
def plot_char(char):
	x_list = [i for i in range(1,30)]
	y_list = [x *2 for x in x_list]   

	# f1 = plt.figure()  
	ax = plt.gca()
	# ax.set_xlabel('x')
	# ax.set_ylabel('y')
	ax.plot(x_list, y_list, color='b', linewidth=1, alpha=0.6)
	ax.scatter(x_list, y_list, color='b',marker = '+', linewidth=1, alpha=0.6)
	plt.show()	

if __name__ == '__main__':
	
	# print(read_dictionary(dic_dir='../data/dic_7356.txt'))
	# plot_char()
	i=10
	if i>20:
		j=2
	print(j)
