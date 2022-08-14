# -*- coding: utf-8 -*-

##
# Copyright (с) Ildar Bikmamatov 2022
# License: MIT
##


from .Model import Model



class Model_2(Model):
	
	
	def __init__(self):
		
		Model.__init__(self)
		
		self.batch_size = 64
		self.epochs = 30
		self.model_name = "model_chars";
		self.train_number = 2;
		
	
	
	"""
		Создаем модель
	"""
	def create(self):
		
		from tensorflow.keras.models import Sequential
		from tensorflow.keras.layers import Dense, Input, Flatten, \
			Dropout, Conv2D, MaxPooling2D, Reshape
		
		model_name = self.get_model_name()
		self.model = Sequential(name=model_name)
		
		input_shape = self.dataset.get_input_shape()
		output_shape = self.dataset.get_output_shape()
		
		# Входной слой
		self.model.add(Input(input_shape, name='input'))
		
		# Reshape
		self.model.add(Reshape( target_shape=(input_shape[0], input_shape[1], 1) ))
		
		# Сверточный слой
		self.model.add(Conv2D(256, kernel_size=(3, 3), padding="same", activation="relu"))
		self.model.add(MaxPooling2D(pool_size=(2, 2)))
		
		# Сверточный слой
		self.model.add(Conv2D(128, kernel_size=(3, 3), padding="same", activation="relu"))
		self.model.add(MaxPooling2D(pool_size=(2, 2)))
		
		# Выравнивающий слой
		self.model.add(Flatten())
		
		# Полносвязные слои
		self.model.add(Dense(512, activation='relu'))
		self.model.add(Dropout(0.5))
		
		# Выходной слой
		self.model.add(Dense(output_shape[0], name='output', activation='softmax'))
		
		# Среднеквадратическая функция ошибки
		self.model.compile(
			loss='mean_squared_error', 
			optimizer='adam',
			metrics=['accuracy'])
	