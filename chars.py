#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

##
# Copyright (с) Ildar Bikmamatov 2022
# License: MIT
##

import torch
from model.CharsNetwork import CharsNetwork
from model.Model2 import Model2


def create_dataset():
	"""
	Создание датасета
	"""
	CharsNetwork.create_chars_dataset()


def check_net(net:CharsNetwork):
	"""
	Проверка модели
	"""
	
	t1 = net.load_train_tensor("0/9|0|28|-10.png")
	t2 = net.load_train_tensor("0/9|0|28|-25.png")
	t3 = net.load_train_tensor("0/9|0|28|-35.png")
	
	tensor = torch.cat(( t1[None,:], t2[None,:], t3[None,:] ))
	
	net.debug(True)
	answer = net.predict(tensor)
	
	#print (answer.shape)
	
	pass


def train_net(net:CharsNetwork):
	"""
	Обучение модели
	"""
	
	# Загрузить сеть с диска
	#net.load()
	net._is_trained = False
	
	# Если модель обучена
	if not net.is_trained():
	
		# Загрузка обучающего датасета
		net.load_dataset("train")
		
		# Обучить сеть
		net.train()
		net.train_show_history()
		
		# Сохранить сеть на диск
		net.save()


def control_net(net:CharsNetwork):
	"""
	Проверка модели
	"""
	
	# Загрузить сеть с диска
	net.load()
	
	# Если модель обучена
	if net.is_loaded():
	
		# Загрузка контрольного датасета
		net.load_dataset("control")
		
		# Проверка модели
		net.control()
	


if __name__ == '__main__':
	
	net = Model2()
	
	# Создать модель
	net.create_model()
	net.summary()
	
	#create_dataset()
	#check_net(net)
	
	#train_net(net)
	control_net(net)
	
	
	pass
