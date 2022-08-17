# -*- coding: utf-8 -*-

##
# Copyright (с) Ildar Bikmamatov 2022
# License: MIT
##

import os, random, math, torch

from typing import Optional
from torch import nn
from torch.nn import Module
from torch.utils.data import DataLoader, TensorDataset, random_split
from ai_helper import *

from .Captcha import generate_captcha_char


# Constant
DATASET_CHARS = "1234567890QWERTYUIOPASDFGHJKLZXCVBNM"
DATASET_CHARS_EX = "1234567890QWERTYUIOPASDFGHJKLZXCVBNMqwertyuiopasdfghjklzxcvbnm"
DATASET_CHARS_COUNT = len(DATASET_CHARS)
DATASET_CHARS_EX_COUNT = len(DATASET_CHARS_EX)


class CharsNetwork(AbstractNetwork):
	
	
	def __init__(self):
		
		AbstractNetwork.__init__(self)
	
	
	def get_name(self):
		
		"""
		Название модели
		"""
		
		return os.path.join("data", "model", "chars", "1")
	
	
	def create_model(self):
		
		"""
		Создает модель
		"""
		
		AbstractNetwork.create_model(self)
		
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
				
				# Выход: 4096
				
				# Полносвязный слои
				x = F.relu(self.fc1(x))
				x = self.drop50(x)
				x = self.fc2(x)
				
				self.net.print_debug("Output:", x.shape)
				
				return x
		
		self.model = Model(self)
		
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)
	
	
	def check_answer(self, **kwargs):
		
		"""
		Check answer
		"""
		
		type = kwargs["type"]
		tensor_y = kwargs["tensor_y"]
		tensor_predict = kwargs["tensor_predict"]
		
		tensor_y = tensor_y.tolist()
		tensor_predict = tensor_predict.round().tolist()
		
		y = get_answer_from_vector(tensor_y)
		predict = get_answer_from_vector(tensor_predict)
		
		if (predict != y) and (type == "control"):
			title = DATASET_CHARS[predict] + " | " + DATASET_CHARS[y]
			#print (title)
			tensor_x = kwargs["tensor_x"]
			tensor_x = tensor_x * 255
			#plot_show_image( tensor_x.tolist(), cmap="gray" )
			pass
		
		return predict == y
	
	
	
	def on_end_epoch(self):
		"""
		Train epoch callback
		"""
		
		epoch_number = self.train_status.epoch_number
		loss_test = self.train_status.loss_test
		acc_train = self.train_status.acc_train
		acc_test = self.train_status.acc_test
		
		if epoch_number >= 25:
			self.stop_training()
		
		if acc_train > 0.95 and epoch_number >= 30:
			self.stop_training()
		
		if acc_test > 0.95 and epoch_number >= 30:
			self.stop_training()
		
		if loss_test < 0.005 and epoch_number >= 50:
			self.stop_training()
	
	
	
	"""	====================== Статические функции ====================== """
	
	
	def get_train_dataset(cls, test_size):
		
		"""
		Возвращает нормализованные обучающие датасеты
		"""
		
		obj = torch.load("data/chars_training.data")
		train_x = obj["x"]
		train_y = obj["y"]
		
		train_dataset = TensorDataset( train_x, train_y )
		test_dataset = cls.get_control_dataset( round(train_x.shape[0] * test_size) )
		
		return train_dataset, test_dataset
	
	
	def get_control_dataset(cls, count = 1000):
		
		"""
		Возвращает нормализованный контрольный датасет
		"""
		
		data_x = torch.tensor([])
		data_y = torch.tensor([])
		
		for i in range(0, count):
		
			char_number = random.randint(0, DATASET_CHARS_EX_COUNT - 1)
			char = DATASET_CHARS_EX[ char_number ]
			
			answer_value = index_of(DATASET_CHARS, char.upper())
			
			angle = random.randint(-50, 50)
			font_size = random.randint(28, 36)
			
			# Генерация изображения
			image = generate_captcha_char(
				char,
				size=font_size,
				angle=angle
			)
			
			x, y = CharsNetwork.get_train_tensor(image, answer_value)
			
			data_x = torch.cat( (data_x, x[None,:]) )
			data_y = torch.cat( (data_y, y[None,:]) )
			
			del x, y
			
		return TensorDataset( data_x, data_y )
	
	
	
	def get_train_tensor(image, char_number = None):
		
		"""
		Возвращает нормализованные обучающие тензора (x, y) по изображению image
		и его номеру char_number
		"""
		
		tensor_y = None
		tensor_x = image_to_tensor(image)
		tensor_x = tensor_x.to(torch.float32) / 255.0
		
		if char_number is not None:
			tensor_y = torch.tensor( get_vector_from_answer(DATASET_CHARS_COUNT)(char_number) )
		
		return tensor_x, tensor_y
	
	
	
	def load_train_tensor(cls, file_path):
		
		"""
		Загружает изображение в тензор из файла
		"""
		
		dir = Directory()
		dir.open("data", "chars")
		
		image = dir.read_file( file_path )
		
		x, _ = CharsNetwork.get_train_tensor(image)
		
		dir.close()
		
		return x
	
	
	
	def create_chars_dataset():
	
		"""
			Создает датасет символов и сохраняет в папку
		"""
		
		dir = Directory()
		dir.open( "data", "chars" )
		
		train_x = torch.tensor([])
		train_y = torch.tensor([])
		
		text_str_count = len(DATASET_CHARS_EX)
		
		angles = [-45,-35,-25,-10,0,10,25,35,45];
		font_sizes = [28,34]
		
		for char_number in range(0, text_str_count):
			
			char = DATASET_CHARS_EX[char_number]
			print (char)
			
			for font_number in range(0, 6):
				for font_size in font_sizes:
					for angle in angles:
					
						image = generate_captcha_char(
							char,
							size=font_size,
							number=font_number,
							angle=angle
						)
						
						file_name = (
							str(char_number) + "|" +
							str(font_number) + "|" +
							str(font_size) + "|" +
							str(angle)
						)
						file_name = os.path.join(
							char.upper(),
							file_name
						)
						
						answer_value = index_of(DATASET_CHARS, char.upper())
						x, y = CharsNetwork.get_train_tensor(image, answer_value)
						
						train_x = torch.cat( (train_x, x[None,:]) )
						data_y = torch.cat( (train_y, y[None,:]) )
						
						dir.save_file(file_name + ".png", image)
						
					pass
			
		obj = {'x': train_x, 'y': train_y}
		torch.save(obj, "data/chars_training.data")
		
		dir.close()
	
	
	def get_char_box(image):
	
		"""
		Находит символ и возвращает прямоугольник вокруг него
		"""
		
		pixels = image.load()
		
		def find_bound(start_pos, direction_x=True, find_start=True):
			
			if direction_x:
			
				for x in range(start_pos, image.size[0]):
					
					count_dot = 0
					for y in range(0, image.size[1]):
						
						color = pixels[x, y]
						if color < 200:
							count_dot = count_dot + 1
					
					if find_start:
						if count_dot >= 1:
							return x
					else:
						if count_dot < 1:
							return x
						
			else:
				
				for y in range(start_pos, image.size[1]):
					
					count_dot = 0
					for x in range(0, image.size[0]):
						
						color = pixels[x, y]
						if color < 200:
							count_dot = count_dot + 1
					
					if find_start:
						if count_dot >= 1:
							return y
					else:
						if count_dot < 1:
							return y
			
			return -1
			
		
		left = find_bound(0, direction_x=True, find_start=True)
		right = find_bound(left + 1, direction_x=True, find_start=False)
		
		top = find_bound(0, direction_x=False, find_start=True)
		bottom = find_bound(top + 1, direction_x=False, find_start=False)
		
		if ((left == -1) or
			(right == -1) or
			(top == -1) or
			(bottom == -1)
		):
			return None
		
		del pixels
		
		left = left - 1
		top = top - 1
		right = right + 1
		bottom = bottom + 1
		
		if right < left: right = image.size[0] - 1
		if bottom < top: bottom = image.size[1] - 1
		
		if left < 0: left = 0
		if top < 0: top = 0
		if right >= image.size[0]: right = image.size[0] - 1
		if bottom >= image.size[1]: bottom = image.size[1] - 1
		
		return (left, top, right, bottom)
		
		
	def normalize_char_image(image, size=(32,32)):
		
		"""
		Находит символ, убирает лишнее побокам пространство,
		и конвертирует в 32x32
		"""
		
		box = CharsNetwork.get_char_box(image)
		
		if box is None:
			return None
			
		image = image.crop( box )
		image = resize_image_canvas(image, size)
		
		return image