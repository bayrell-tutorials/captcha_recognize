#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

##
# Copyright (с) Ildar Bikmamatov 2022
# License: MIT
##

import os, random

from ai_helper import *
from .CharsNetwork import CharsNetwork
#from .ModelTorch1 import ModelTorch1


# Constant
DATASET_CHARS = "1234567890QWERTYUIOPASDFGHJKLZXCVBNM"
DATASET_CHARS_EX = "1234567890QWERTYUIOPASDFGHJKLZXCVBNMqwertyuiopasdfghjklzxcvbnm"
DATASET_CHARS_COUNT = len(DATASET_CHARS)
DATASET_CHARS_EX_COUNT = len(DATASET_CHARS_EX)


# Known models
model_list = [
	CharsNetwork(),
]


def get_model(model_name):
	
	"""
		Возратит модель
	"""
	
	for model in model_list:
		if model.model_name == model_name:
			return model
			
	return None
	
	
def get_captcha_path(photo_number):
	
	"""
		Возвращает путь к капче по ее номеру
	"""
	
	parent_image_dir = str(photo_number % 100)
	parent_image_dir = parent_image_dir.zfill(2)
	file_name = str(photo_number)
	
	image_result_path = os.path.join(parent_image_dir, file_name) + ".png"
	image_mask_path = os.path.join(parent_image_dir, file_name) + "-mask.png"
	image_json_path = os.path.join(parent_image_dir, file_name) + "-json.txt"
	
	return {
		"dir": os.path.join(parent_image_dir),
		"image": image_result_path,
		"mask": image_mask_path,
		"json": image_json_path,
	}
	
	
	
def read_captcha(data_stream, photo_number):
		
	"""
		Загружает капчу из файла по ее номеру
	"""
	
	from .Captcha import Captcha
	from .lib import get_captcha_path
	
	captcha = None
	path = get_captcha_path(photo_number)
	
	try:
		captcha = Captcha()
		captcha.image = data_stream.read_file(path["image"])
		captcha.mask = data_stream.read_file(path["mask"])
		image_json = data_stream.read_file(path["json"])
		image_json = image_json.decode("utf-8")
		captcha.load_json(image_json)
		
		if not captcha.is_load():
			captcha = None
		
	except Exception:
		captcha = None
	
	return captcha
	
	
	
def save_captcha(data_stream, photo_number, captcha):
	
	"""
		Сохраняет капчу в файл по ее номеру
	"""
	
	from .lib import get_captcha_path
	
	path = get_captcha_path(photo_number)
	
	data_stream.save_file(path["image"], captcha.image)
	data_stream.save_file(path["mask"], captcha.mask)
	data_stream.save_file(path["json"], captcha.get_json())
	
	
	
def get_train_vector_chars(image, char_number):
	
	"""
		Возвращает обучающий вектор (x, y) по изображению image и его номеру char_number
	"""
	
	question_vector = image_to_tensor(image)
	question_vector = question_vector.astype('float32') / 255.0
	
	answer_vector = get_vector_from_answer(DATASET_CHARS_COUNT)(char_number)
	
	return question_vector, answer_vector
	
	
	
def image_get_char_box(image):
	
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
	
	
def image_char_normalize(image, size=(32,32)):
	
	"""
		Находит символ, убирает лишнее побокам пространство,
		и конвертирует в 32x32
	"""
	
	box = image_get_char_box(image)
	
	if box is None:
		return None
		
	image = image.crop( box )
	image = resize_image_canvas(image, size)
	
	return image