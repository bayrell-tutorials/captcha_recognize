# -*- coding: utf-8 -*-

##
# Copyright (с) Ildar Bikmamatov 2022
# License: MIT
##

import os,  torch

from torch import nn
from model.CharsNetwork import DATASET_CHARS_COUNT, CharsNetwork


class Model2(CharsNetwork):
	
	
	def get_name(self):
		
		"""
		Название модели
		"""
		
		return os.path.join("data", "model", "chars", "3")
	
	
	def create_model(self):
		
		"""
		Создает модель
		"""
		
		CharsNetwork.create_model(self)
		
		import torch.nn.functional as F
		
		class Model(nn.Module):
			def __init__(self, net):
				super(Model, self).__init__()
				
				self.net = net
				self.input_shape = (32, 32, 3)
				self.output_shape = (10)
				
				# Дополнительные слои
				self.max_pool = nn.MaxPool2d(2, 2)
				self.drop25 = nn.Dropout(0.25)
				self.drop50 = nn.Dropout(0.50)
				
				# Сверточный слой
				self.conv1 = nn.Conv2d(1, 128, kernel_size=3, padding=(1,1))
				self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=(1,1))
				
				# Полносвязный слой
				self.fc1 = nn.Linear(4096, 256)
				self.fc2 = nn.Linear(256, DATASET_CHARS_COUNT)
			
			
			def forward(self, x):
				
				x = x[:,None,:]
				
				self.net.print_debug("Input:", x.shape)
				
				# Сверточный слой 1
				# Вход: 1, 32, 32
				x = F.relu(self.conv1(x))
				self.net.print_debug("Conv1:", x.shape)
				
				# Выход: 128, 32, 32
				
				# Макс пулинг
				x = self.max_pool(x)
				self.net.print_debug("Max pool1:", x.shape)
				
				# Выход: 128, 16, 16
				
				# Сверточный слой 2
				x = F.relu(self.conv2(x))
				self.net.print_debug("Conv2:", x.shape)
				
				# Выход: 64, 16, 16
				
				# Макс пулинг
				x = self.max_pool(x)
				self.net.print_debug("Max pool2:", x.shape)
				
				# Выход: 64, 8, 8
				
				# Выравнивающий слой
				x = x.view(-1, 4096)
				self.net.print_debug("Line:", x.shape)
				
				# Выход: 4096 => 256 => 36
				
				x = self.drop50(x)
				
				# Полносвязный слой
				x = F.relu(self.fc1(x))
				x = self.drop50(x)
				x = self.fc2(x)
				
				#x = F.softmax(x, dim=0)
				#x = F.sigmoid(x)
				#x = torch.sigmoid(x)
				
				self.net.print_debug("Output:", x.shape)
				
				return x
		
		self.model = Model(self)
		
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)
		#self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
		self.loss = nn.CrossEntropyLoss()
		
	