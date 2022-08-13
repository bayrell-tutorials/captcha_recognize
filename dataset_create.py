#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

##
# Copyright (с) Ildar Bikmamatov 2022
# License: MIT
##

from lib import *


def generate_captcha_dataset():

	dataset = DataSet()
	dataset.open("data/captcha_dataset.zip")

	count_images = 1000
	for i in range(0, count_images):
		print (i)
		dataset.generate_captcha(i, force=False)

	dataset.close()


def generate_captcha_symbols():
	
	dataset = DataSet()
	dataset.open("data/captcha_dataset.zip")
	
	text_str="1234567890QWERTYUIOPASDFGHJKLZXCVBNMqwertyuiopasdfghjklzxcvbnm"
	text_str_count = len(text_str)
	
	angles = [
		-45
		-25,
		-10,
		0,
		10,
		25,
		45
	];
	
	font_sizes = [
		28,
		34
	]
	
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
						"chars",
						char.upper(),
						file_name
					)
					
					#print (file_name)
					dataset.save_file(file_name + ".png", image)
					
					#return
					
				pass
		
	
	dataset.close()
	

def generate_captcha_and_show():
	captcha = Captcha()
	captcha.generate()
	captcha.resize_max()
	captcha.show()


def dataset_generate_symbol():
	captcha = Captcha()
	
	import PIL.ImageOps
	
	font = captcha.get_font(size=28, number=1)
	angle = -45
	
	image = captcha.get_rotated_text("W", font, angle)
	image = PIL.ImageOps.invert(image)
	image = image_symbol_normalize(image)
	
	plt.imshow(image, cmap='gray')
	plt.show()


def dataset_test():
	
	dataset = DataSet()
	dataset.open("data/captcha_dataset.zip")
	
	captcha = dataset.get_captcha(1)
	print (captcha)
	
	dataset.close();


#generate_captcha_dataset()
#generate_captcha_symbols()
#generate_captcha_and_show()
#dataset_generate_symbol()
#dataset_test()
