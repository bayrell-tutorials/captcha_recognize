# -*- coding: utf-8 -*-

##
# Copyright (с) Ildar Bikmamatov 2022
# License: MIT
##

import math, random, json
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont, ImageOps
from .Image import image_resize_canvas, image_symbol_normalize


class Captcha:
	
	def __init__(self):
		self.image = None
		self.mask = None
		self.answer = None
		self.words = None
		self.image = None
		self.image_width_max = 400
		self.image_height_max = 150
		self.image_width = random.randint(300, self.image_width_max)
		self.image_height = random.randint(90, self.image_height_max)
		self.text_count = random.randint(4, 6)
		self.dots_count = random.randint(10, 20)
		self.lines_count = random.randint(2, 3)
	
	
	def is_load(self):
		return (
			self.image is not None and
			self.mask is not None and
			self.answer is not None and
			self.words is not None
		)
	
	
	def load_json(self, json_string):
		try:
			obj = json.loads(json_string)
			self.answer = obj["answer"]
			self.words = obj["words"]
		except Exception:
			self.answer = None
			self.words = None
	
	
	def get_json(self):
		obj = {
			"answer": self.answer,
			"words": self.words,
		}
		
		json_string = json.dumps(obj)
		return json_string
	
	
	def show(self):
		
		if self.image is None or self.mask is None:
			return
		
		image = self.image.copy()
		image_mask = self.mask.convert("RGB")
		image_mask_draw = ImageDraw.Draw(image_mask)
		
		image_center = (
			math.ceil(image_mask.size[0] / 2),
			math.ceil(image_mask.size[1] / 2)
		)
		
		for word in self.words:
			text_box = word["box"]
			
			text_box = (
				text_box[0] + image_center[0],
				text_box[1] + image_center[1],
				text_box[2], text_box[3]
			)
			
			image_mask_draw.rectangle(
				(
					text_box[0] - 1,
					text_box[1] - 1,
					text_box[0] + text_box[2] + 1,
					text_box[1] + text_box[3] + 1
				),
				outline="green",
				width=2
			)
		
		#print (self.words)
		
		rows = 2
		columns = 1
		
		#image_mask = self.mask.copy()
		#image_mask = self.mask.convert("1")
		
		fig = plt.figure()
		
		fig.add_subplot(rows, columns, 1)
		plt.imshow(image, cmap='gray')
		plt.title(
			"Size: " +
			str(image.size[0]) + "x" + str(image.size[1]) +
			". Answer: " + self.answer.upper())
		
		fig.add_subplot(rows, columns, 2)
		plt.imshow(image_mask, cmap='gray')
		
		plt.savefig("tmp/captcha.png")
		plt.show()
		
		del image, image_mask
	
	
	def get_font(self, size=28, number=-1):
			
		font_arr = [
			"fonts/Roboto/Roboto-Bold.ttf",
			"fonts/Roboto/Roboto-Regular.ttf",
			"fonts/Roboto/Roboto-Italic.ttf",
			"fonts/Montserrat/static/Montserrat-Bold.ttf",
			"fonts/Montserrat/static/Montserrat-Italic.ttf",
			"fonts/Montserrat/static/Montserrat-Regular.ttf",
		]
		
		if number <= 0:
			number = random.randint(0, len(font_arr) - 1)
		
		font_path = font_arr[ number ]
		font = ImageFont.truetype(font_path, size)
		
		return font
	
	
	def create_image(self, mode="RGB", color="white"):
		img = Image.new(mode, (self.image_width, self.image_height), color = color)
		draw = ImageDraw.Draw(img)
		return (img, draw)
	
	
	"""
		Рисует линии
	"""
	def draw_line(self, draw, color, width):
			
		padding_x = 25
		padding_y = 10
		
		position = (
			random.randint(padding_x, padding_x),
			random.randint(padding_y, self.image_height - padding_y),
			random.randint(self.image_width - padding_x, self.image_width - padding_x),
			random.randint(padding_y, self.image_height - padding_y)
		)
		
		draw.line(position, fill=color, width=width)
	
	
	"""
		Рисует точки
	"""
	def draw_dots(self, draw, color, width):
		
		padding_x = 25
		padding_y = 10
		x = random.randint(padding_x, self.image_width - padding_x)
		y = random.randint(padding_y, self.image_height - padding_y)
		
		position = (x,y,x+width,y+width)
		draw.ellipse(position, fill=color, width=width)
	
	
	"""
		Рисует текст капчи
	"""
	def get_rotated_text(self, text, font, angle):
		
		text_size = font.getsize(text)
		image_size = (text_size[1] + 2, text_size[1] + 2)
		
		if text_size[0] > text_size[1]:
			image_size = (text_size[0] + 2, text_size[0] + 2)
		
		img = Image.new('L', image_size, color="black")
		
		draw = ImageDraw.Draw(img)
		draw.text(
			(
				math.ceil((image_size[0] - text_size[0]) / 2),
				math.ceil((image_size[1] - text_size[1]) / 2),
			),
			text,
			"white",
			font=font
		)
		
		img = img.rotate(
			angle,
			expand=1,
			center=(
				math.ceil(image_size[0] / 2),
				math.ceil(image_size[1] / 2)
			)
		)
		
		del draw
		
		return img
		
	
	"""
		Рисует текст капчи
	"""
	def draw_text(self, image_result, image_mask, position_center, text, color, font, angle):
		
		img = self.get_rotated_text(text, font, angle)
		img_size = img.size
		
		position = (
			math.ceil(position_center[0] - img_size[0] / 2),
			math.ceil(position_center[1] - img_size[1] / 2)
		)
		
		if image_result is not None:
			image_result.paste(
				ImageOps.colorize(img, black=(0, 0, 0), white=color),
				position,
				img
			)
			
		if image_mask is not None:
			image_mask.paste(
				ImageOps.colorize(img, black=(0, 0, 0), white="black"),
				position,
				img
			)
		
		del img
		
		return (
			position[0],
			position[1],
			img_size[0],
			img_size[1]
		)
	
	
	
	"""
		Алгоритм генерации капчи
	"""
	def generate(self):
		
		text_str="1234567890QWERTYUIOPASDFGHJKLZXCVBNMqwertyuiopasdfghjklzxcvbnm"
		text_str_count = len(text_str)
		image_result = None
		draw_result = None
		image_mask = None
		draw_mask = None
		image_words = []
		
		colors = [
			(255,255,204),
			(255,255,153),
			(255,255,102),
			(255,255,51),
			(255,153,51),
			(255,204,153),
			(255,255,0),
			(255,153,0),
			(255,102,51),
			(255,204,204),
			(255,204,255),
			(255,153,255),
			(255,102,255),
			(255,51,255),
			(255,0,255),
			(255,153,153),
			(255,102,204),
			(255,0,204),
			(255,51,204),
			(204,102,255),
			(204,51,255),
			(204,153,255),
			(204,204,255),
			(153,153,255),
			(153,153,204),
			(102,153,255),
			(102,153,204),
			(153,204,255),
			(51,153,255),
			(204,255,255),
			(153,255,255),
			(102,255,255),
			(51,255,255),
			(102,204,153),
			(204,204,204),
			(204,255,204),
			(51,204,204),
		]
		
		self.image_color = colors[ random.randint(0, len(colors) - 1) ]
		image_result, draw_result = self.create_image( mode="RGB", color=self.image_color )
		image_mask, draw_mask = self.create_image(mode="L", color="white")
		
		image_center = (
			math.ceil(self.image_width / 2),
			math.ceil(self.image_height / 2)
		)
		
		text_answer = ""
		step_text = self.image_width / self.text_count
		step_text_x = math.ceil(step_text / 4)
		step_text_y = math.ceil(self.image_height / 5)
		for i in range(0, self.text_count):
			
			pos_x = step_text * i + step_text / 2
			pos_y = self.image_height / 2
			
			pos_x += random.randint(-step_text_x, step_text_x)
			pos_y += random.randint(-step_text_y, step_text_y)
			
			angle = random.randint(-50, 50)
			text = text_str[ random.randint(0, text_str_count - 1) ]
			text_answer += text
			font = self.get_font( size=random.randint(28, 36) )
			
			color = (
				random.randint(10, 200),
				random.randint(10, 200),
				random.randint(10, 200),
			)
			
			text_box = self.draw_text(
				position_center=(pos_x, pos_y),
				text=text,
				color=color,
				font=font,
				angle=angle,
				image_mask=image_mask,
				image_result=image_result
			)
			
			text_box = (
				text_box[0] - image_center[0],
				text_box[1] - image_center[1],
				text_box[2], text_box[3]
			)
			#print(text_box)
			image_words.append({
				"box": text_box,
				"text": text,
			})
			
			del font
		
		
		for i in range(0, self.lines_count):
			color = (
				random.randint(10, 200),
				random.randint(10, 200),
				random.randint(10, 200),
			)
			self.draw_line(
				color=color,
				width=random.randint(2, 3),
				draw=draw_result
			)
		
		
		for i in range(0, self.dots_count):
			color = (
				random.randint(10, 200),
				random.randint(10, 200),
				random.randint(10, 200),
			)
			self.draw_dots(
				color=color,
				width=random.randint(3, 5),
				draw=draw_result
			)
		
		del draw_result
		del draw_mask
		
		self.image = image_result
		self.mask = image_mask
		self.answer = text_answer
		self.words = image_words
		

	def resize_max(self):
		
		self.image = image_resize_canvas(self.image, self.image_width_max, self.image_height_max)
		self.mask = image_resize_canvas(self.mask, self.image_width_max, self.image_height_max)
		
		pass
	
	
def generate_captcha_char(char, size=28, number=1, angle=0):
	
	captcha = Captcha()
	font = captcha.get_font(size=size, number=number)
	image = captcha.get_rotated_text(char, font, angle)
	image = ImageOps.invert(image)
	image = image_symbol_normalize(image)

	return image