#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

##
# Copyright (с) Ildar Bikmamatov 2022
# License: MIT
##

import os

from lib.Captcha import generate_captcha_char
from lib.DataSet import DataSetReader


def generate_captcha_dataset():
	
	dataset_stream = DataSetReader()
	dataset_stream.open("data_chars")

	count_images = 1000
	for i in range(0, count_images):
		print (i)
		dataset_stream.generate_captcha(i, force=False)
	
	dataset_stream.close()



def generate_captcha_symbols():
	
	dataset_stream = DataSetReader()
	dataset_stream.open("data_chars")
	
	text_str="1234567890QWERTYUIOPASDFGHJKLZXCVBNMqwertyuiopasdfghjklzxcvbnm"
	text_str_count = len(text_str)
	
	angles = [-45,-35,-25,-10,0,10,25,35,45];
	font_sizes = [28,34]
	
	for char_number in range(0, text_str_count):
		
		char = text_str[char_number]
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


generate_captcha_symbols()
