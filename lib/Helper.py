# -*- coding: utf-8 -*-

##
# Copyright (с) Ildar Bikmamatov 2022
# License: MIT
##

import numpy as np


"""
	Математическая функция Sign определения знаказ числа
"""
def sign(x):
	if x >= 0: return 1
	return -1
	
	
"""
	Поиск значения в массиве
"""
def indexOf(arr, item):
	try:
		index = arr.index(item)
		return index
	except Exception:
		pass
	return -1
	
	
"""
	Конкатенаци векторов
"""
def vector_append(res, data):
	
	if res is None:
		res = np.expand_dims(data, axis=0)
	else:
		res = np.append(res, [data], axis=0)
	
	return res
	
	
# Init tensorflow
def tensorflow_gpu_init(memory_limit=1024):
	import tensorflow as tf
	gpus = tf.config.list_physical_devices('GPU')
	tf.config.experimental.set_memory_growth(gpus[0], True)
	tf.config.experimental.set_virtual_device_configuration(
	    gpus[0],
	    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])
