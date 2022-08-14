#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

##
# Copyright (с) Ildar Bikmamatov 2022
# License: MIT
##

import os, random

from ai_helper import *
from model.Captcha import Captcha, generate_captcha_char
from model.lib import DATASET_CHARS, DATASET_CHARS_COUNT, \
	DATASET_CHARS_EX, DATASET_CHARS_EX_COUNT, get_train_vector_chars


def create_captcha_dataset():
	
	"""
		Создает датасет капчи
	"""
	
	dataset_stream = DataStream()
	dataset_stream.open("data", "captcha")
	
	force = False
	count_images = 1000
	for photo_number in range(0, count_images):
		
		print (photo_number)
		
		# Проверка есть ли такой уже файл
		if not force:
			captcha = dataset_stream.read_captcha(photo_number)
			if captcha is not None:
				del captcha
				continue
		
		captcha = Captcha()
		captcha.generate()
		captcha.resize_max()
		
		dataset_stream.save_captcha(photo_number, captcha)
		
		del captcha
	
	dataset_stream.close()



def create_chars_dataset():
	
	"""
		Создает датасет символов и сохраняет в папку
	"""
	
	dataset_stream = DataStream()
	dataset_stream.open( "data", "chars" )
	
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
					
					#print (file_name)
					dataset_stream.save_file(file_name + ".png", image)
					
					#return
					
				pass
		
	
	dataset_stream.close()


def load_train_dataset_chars():
	
	"""
		Возвращает датасет для распознования букв.
		Загружает данные из папки data/chars
	"""
	
	dataset = DataSet()
	
	dataset_reader = DataStream()
	dataset_reader.open("data", "chars")
	
	for char_number in range(0, DATASET_CHARS_COUNT):
		
		char = DATASET_CHARS[ char_number ]
		files = dataset_reader.list_files(char)
		
		for file in files:
			
			# Получаем изображение
			image = dataset_reader.read_file( os.path.join(char, file) )
			
			x, y = get_train_vector_chars(image, char_number)
			dataset.append(x, y)
	
	return dataset


def create_random_chars_dataset(count=1000):
	
	"""
		Возвращает рандомный датасет
	"""
	
	dataset = DataSet()
	
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
		
		x, y = get_train_vector_chars(image, answer_value)
		dataset.append(x, y)
	
	
	return dataset


if __name__ == '__main__':
	print ("Create dataset")
	#create_captcha_dataset()
	#create_chars_dataset()
	pass
