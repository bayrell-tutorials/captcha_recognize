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
		
		self.epochs = 5
		
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
				self.conv1 = nn.Conv2d(1, 128, kernel_size=5)
				self.conv2 = nn.Conv2d(128, 64, kernel_size=5)
				
				# Полносвязный слой
				self.fc1 = nn.Linear(1600, 256)
				self.fc2 = nn.Linear(256, DATASET_CHARS_COUNT)
			
			
			def forward(self, x):
				
				x = x[:,None,:]
				
				self.net.print_debug("Input:", x.shape)
				
				# Сверточный слой 1
				# Вход: 1, 32, 32
				x = F.relu(self.conv1(x))
				self.net.print_debug("Conv1:", x.shape)
				
				# Выход: 128, 28, 28
				
				# Макс пулинг
				x = self.max_pool(x)
				self.net.print_debug("Max pool1:", x.shape)
				
				# Выход: 128, 14, 14
				
				# Сверточный слой 2
				x = F.relu(self.conv2(x))
				self.net.print_debug("Conv2:", x.shape)
				
				# Выход: 64, 10, 10
				
				# Макс пулинг
				x = self.max_pool(x)
				self.net.print_debug("Max pool2:", x.shape)
				
				# Выход: 64, 5, 5
				
				# Выравнивающий слой
				x = x.view(-1, 1600)
				self.net.print_debug("Line:", x.shape)
				
				# Выход: 1600
				
				# Полносвязный слои
				x = F.relu(self.fc1(x))
				x = self.drop50(x)
				x = F.softmax(self.fc2(x), dim=0)
				
				self.net.print_debug("Layer3:", x.shape)
								
				return x
		
		self.model = Model(self)
	
	
	
	def check_answer(self, **kwargs):
		
		"""
		Check answer
		"""
		
		tensor_y = kwargs["tensor_y"]
		tensor_predict = kwargs["tensor_predict"]
		
		tensor_y = tensor_y.tolist()
		tensor_predict = tensor_predict.round().tolist()
		
		y = get_answer_from_vector(tensor_y)
		predict = get_answer_from_vector(tensor_predict)
		
		if (predict != y) and (type == "control"):
			#title = DATASET_CHARS[predict] + " | " + DATASET_CHARS[y]
			#print (title)
			pass
		
		return predict == y
	
	
	
	def train_epoch_callback(self, **kwargs):
		"""
		Train epoch callback
		"""
		
		epoch_number = kwargs["epoch_number"]
		loss_test = kwargs["loss_test"]
		accuracy_train = kwargs["accuracy_train"]
		accuracy_test = kwargs["accuracy_test"]
		
		if epoch_number >= 50:
			self.stop_training()
		
		if accuracy_train > 0.95:
			self.stop_training()
		
		if accuracy_test > 0.95:
			self.stop_training()
		
		if loss_test < 0.005 and epoch_number >= 5:
			self.stop_training()
	
	
	
	"""	====================== Статические функции ====================== """
	
	
	def get_train_dataset(cls, test_size):
		
		"""
		Возвращает нормализованные обучающие датасеты
		"""
		
		from sklearn.model_selection import train_test_split
		
		dir = Directory()
		dir.open("data", "chars")
		
		train_x = torch.tensor([])
		train_y = torch.tensor([])
		test_x = torch.tensor([])
		test_y = torch.tensor([])
		
		for char_number in range(0, DATASET_CHARS_COUNT):
		
			char = DATASET_CHARS[ char_number ]
			files = dir.list_files(char)
			
			answer_value = indexOf(DATASET_CHARS, char.upper())
			
			data_arr = []
			
			for file in files:
				
				# Получаем изображение
				image = dir.read_file( os.path.join(char, file) )
				
				x, y = CharsNetwork.get_train_tensor(image, answer_value)
				
				data_arr.append((x, y))
				
				del x, y
			
			# Разделяем данные на обучающие и тестовые
			# В датасете будет одинаковы процент тестовых данных для каждой буквы
			train, test = train_test_split(data_arr, test_size=test_size)
			
			for item in train:
				train_x = torch.cat((train_x, item[0][None,:]))
				train_y = torch.cat((train_y, item[1][None,:]))
			
			for item in test:
				test_x = torch.cat((test_x, item[0][None,:]))
				test_y = torch.cat((test_y, item[1][None,:]))
			
			#break
			
		train_dataset = TensorDataset( train_x, train_y )
		test_dataset = TensorDataset( test_x, test_y )
		
		return train_dataset, test_dataset
	
	
	def get_control_dataset(cls):
		
		"""
		Возвращает нормализованный контрольный датасет
		"""
		
		count = 1000
		
		data_x = torch.tensor([])
		data_y = torch.tensor([])
		
		for i in range(0, count):
		
			char_number = random.randint(0, DATASET_CHARS_EX_COUNT - 1)
			char = DATASET_CHARS_EX[ char_number ]
			
			answer_value = indexOf(DATASET_CHARS, char.upper())
			
			angle = random.randint(-50, 50)
			font_size = random.randint(28, 36)
			
			# Генерация изображения
			image = generate_captcha_char(
				char,
				size=font_size,
				angle=angle
			)
			
			x, y = CharsNetwork.get_train_tensor(image, answer_value)
			
			data_x = torch.cat((data_x, x[None,:]))
			data_y = torch.cat((data_y, y[None,:]))
			
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
						
						dir.save_file(file_name + ".png", image)
						
					pass
			
		
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
		
		
	def char_image_normalize(image, size=(32,32)):
		
		"""
		Находит символ, убирает лишнее побокам пространство,
		и конвертирует в 32x32
		"""
		
		box = CharsNetwork.get_char_box(image)
		
		if box is None:
			return None
			
		image = image.crop( box )
		image = image_resize_canvas(image, size)
		
		return image