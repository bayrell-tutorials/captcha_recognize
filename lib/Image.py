# -*- coding: utf-8 -*-

##
# Copyright (с) Ildar Bikmamatov 2022
# License: MIT
##

import math, io
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


"""
	Изменение размеров холста картинки
"""
def image_resize_canvas(image, width, height):
	
	pixels = image.load()
	
	image_new = Image.new(image.mode, (width, height), color = pixels[0, 0])
	draw = ImageDraw.Draw(image_new)
	
	position = (
		math.ceil((width - image.size[0]) / 2),
		math.ceil((height - image.size[1]) / 2),
	)
	
	image_new.paste(image, position)
	
	del pixels, draw, image
	
	return image_new
	

"""
	Находит символ и возвращает прямоугольник вокруг него
"""
def image_get_symbol_box(image):
	
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
	
	
"""
	Находит символ и убирает лишнее побокам пространство
"""
def image_symbol_normalize(image, width=32, height=32):
	
	box = image_get_symbol_box(image)
	
	if box is None:
		return None
		
	image = image.crop( box )
	image = image_resize_canvas(image, width, height)
	
	return image
	
	
"""
	Преобразует картинку в вектор
"""
def image_to_vector(image_bytes, mode=None):
	
	image = None
	
	try:
		
		if isinstance(image_bytes, bytes):
			image = Image.open(io.BytesIO(image_bytes))
		
		if isinstance(image_bytes, Image.Image):
			image = image_bytes
	
	except Exception:
		image = None
	
	if image is None:
		return None
	
	if mode is not None:
		image = image.convert(mode)
	
	image_vector = np.asarray(image)

	return image_vector