# -*- coding: utf-8 -*-

##
# Copyright (с) Ildar Bikmamatov 2022
# License: MIT
##

import os

from ai_helper import *



class ModelKeras1(KerasModel):
	
	
	def __init__(self):
		
		KerasModel.__init__(self)
		
		self.batch_size = 64
		self.epochs = 30
		self.model_name = os.path.join("data", "model", "keras1")
		
	
	
	"""
		Создаем модель
	"""
	def create(self):
		
		from tensorflow.keras.models import Sequential
		from tensorflow.keras.layers import Dense, Input, Flatten, \
			Dropout, Conv2D, MaxPooling2D, MaxPooling3D, Reshape
		
		model_name = self.get_model_name()
		self.model = Sequential(name=model_name)
		
		# Входной слой
		self.model.add(Input(self.input_shape, name='input'))
		
		# Reshape
		self.model.add(Reshape( target_shape=(self.input_shape[0], self.input_shape[1], 1) ))
		
		# Сверточный слой
		self.model.add(Conv2D(128, kernel_size=(3, 3), padding="same", activation="relu"))
		self.model.add(MaxPooling2D(pool_size=(2, 2)))
		
		# Сверточный слой
		self.model.add(Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"))
		self.model.add(MaxPooling2D(pool_size=(2, 2)))
		
		# Выравнивающий слой
		self.model.add(Flatten())
		
		# Полносвязные слои
		self.model.add(Dense(256, activation='relu'))
		self.model.add(Dropout(0.5))
		
		# Выходной слой
		self.model.add(Dense(self.output_shape, name='output', activation='softmax'))
		
		# Среднеквадратическая функция ошибки
		self.model.compile(
			loss='mean_squared_error', 
			optimizer='adam',
			metrics=['accuracy'])
			
		self._is_new = True
	