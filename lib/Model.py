# -*- coding: utf-8 -*-

##
# Copyright (с) Ildar Bikmamatov 2022
# License: MIT
##


import os, math

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt

from .DataSet import DATASET_DIR


class Model:
	
	def __init__(self):
		
		self.model = None
		self.history = None
		self.batch_size = 64
		self.epochs = 30
		self.dataset = None
		self.model_name = "model";
		self.train_number = 1;
		
	
	def get_model_name(self):
		return self.model_name + "_" + str(self.train_number)
	
	
	def get_model_path(self):
		return os.path.join(DATASET_DIR, self.model_name, str(self.train_number))
	
	
	def is_loaded(self):
		return self.model != None
	
	
	def set_dataset(self, dataset):
		self.dataset = dataset
		self.dataset.build()
	
	
	"""
		Создаем модель
	"""
	def create(self):
		
		from tensorflow.keras.models import Sequential
		from tensorflow.keras.layers import Dense, Input, Dropout
		
		model_name = self.get_model_name()
		self.model = Sequential(name=model_name)
		
		input_shape, output_shape, _, _ = self.dataset.get_shape()
		
		# Входной слой
		self.model.add(Input(input_shape, name='input'))
		
		# Полносвязные слои
		self.model.add(Dense(256, activation='relu'))
		self.model.add(Dropout(0.5))
		
		# Выходной слой
		self.model.add(Dense(output_shape, name='output', activation='softmax'))
		
		# Среднеквадратическая функция ошибки
		self.model.compile(
			loss='mean_squared_error', 
			optimizer='adam',
			metrics=['accuracy'])
		
		
	"""
		Создать папку для модели
	"""
	def create_model_parent_dir(self):
		
		# Create model folder
		model_path = self.get_model_path()
		model_dir = os.path.dirname(model_path)
		
		if not os.path.isdir(model_dir):
			os.makedirs(model_dir)
		
		
	"""
		Загружает модель
	"""
	def load(self):
		
		model_path = self.get_model_path()
		model_file_path = model_path + '.h5'
		
		self.model = None
		if os.path.isfile(model_file_path):
			self.model = keras.models.load_model(model_file_path)
		
		
	"""
		Показывает информацию о модели
	"""
	def show(self):
		
		self.create_model_parent_dir()
		model_path = self.get_model_path()
		
		# Вывод на экран информация о модели
		self.model.summary()
		
		file_name = model_path + "_plot.png"
		
		keras.utils.plot_model(
			self.model,
			to_file=file_name,
			show_shapes=True)
		
		
	"""
		Обучение нейронной сети
	"""
	def train(self):
		
		model_path = self.get_model_path()
		checkpoint_path = os.path.join(model_path, "training", "cp.ckpt")
		checkpoint_dir = os.path.dirname(checkpoint_path)
		
		# Создаем папку, куда будут сохраняться веса во время обучения
		if not os.path.isdir(checkpoint_dir):
			os.makedirs(checkpoint_dir)
		
		# Callback функия для сохранения весов
		cp_callback = keras.callbacks.ModelCheckpoint(
			filepath=checkpoint_path,
			save_weights_only=True,
			verbose=1
		)
		
		self.dataset.build()
		
		self.history = self.model.fit(
		
			# Входные данные
			self.dataset.train_x,
			
			# Выходные данные
			self.dataset.train_y,
			
			# Размер партии для обучения
			batch_size=self.batch_size,
			
			# Количество эпох обучения
			epochs=self.epochs,
			
			# Контрольные данные
			validation_data=(self.dataset.control_x, self.dataset.control_y),
			
			# Подробный вывод
			verbose=1,
			
			# Сохраняем контрольные точки
			callbacks=[cp_callback]
		) 
		
		# Сохраняем модель на диск
		self.model.save(model_path)
		self.model.save(model_path + ".h5")
		
		pass
	
	
	"""
		Показывает как обучилась нейронная сеть
	"""
	def train_show(self):
		
		model_path = self.get_model_path()
		total_val_accuracy = math.ceil(self.history.history['val_accuracy'][-1] * 100)
	
		# Сохраняем картинку
		plt.title("Итог: " + str(total_val_accuracy) + "%")
		plt.plot( np.multiply(self.history.history['accuracy'], 100), label='Обучение')
		plt.plot( np.multiply(self.history.history['val_accuracy'], 100), label='Контрольные ответы')
		plt.plot( np.multiply(self.history.history['val_loss'], 100), label='Ошибка')
		plt.ylabel('Процент')
		plt.xlabel('Эпоха')
		plt.legend()
		plt.savefig(model_path + '_history.png')
		plt.show()
		
	
	
	"""
		Проверка модели
	"""
	def check(self, test_dataset, callback=None):
		
		vector_x = test_dataset.get_x()
		vector_y = test_dataset.get_y()
		
		# Спрашиваем модель
		vector_answer = self.model.predict( vector_x )
		
		# Выводим ответы
		correct_answers = 0
		total_questions = len(vector_x)
		
		for i in range(0, total_questions):
			
			if callback != None:
				correct = callback(
					question = vector_x[i],
					answer = vector_answer[i],
					control = vector_y[i],
				)
				if correct:
					correct_answers = correct_answers + 1
		
		return correct_answers, total_questions
	
	