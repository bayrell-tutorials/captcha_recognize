#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

##
# Copyright (с) Ildar Bikmamatov 2022
# License: MIT
##

import os

from ai_helper import *
from model.Captcha import Captcha, generate_captcha_char
from model.DataReader import DataReader
from model.lib import DATASET_CHARS_EX


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



def create_symbols_dataset():
	
	"""
		Создает датасет символов
	"""
	
	dataset_stream = DataReader()
	dataset_stream.open( "data", "dataset" )
	
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


#create_captcha_dataset()
create_symbols_dataset()