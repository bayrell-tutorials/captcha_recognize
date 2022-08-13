# -*- coding: utf-8 -*-

##
# Copyright (с) Ildar Bikmamatov 2022
# License: MIT
##

from cgitb import text
import io, os, random, math, zipfile, shutil, json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont, ImageOps


# Random seed
random.seed()


"""
	Математическая функция Sign определения знаказ числа
"""
def sign(x):
	if x >= 0: return 1
	return -1


"""
	Поиск значения в массиве
"""
def indexOf(arr, item):
	try:
		index = arr.index(item)
		return index
	except Exception:
		pass
	return -1


"""
	Преобразует картинку в вектор
"""
def convert_image_to_vector(image_bytes, is_rgb=True):
	
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
	
	if not is_rgb:
		image = image.convert("L")

	image_vector = np.asarray(image)

	return image_vector


"""
	Загружает файл и преобразует его в вектор
"""
def load_image_as_vector(file_path):
	
	if not os.path.isfile(file_path):
		return None
	
	image_vector = None
	
	try:
		image = Image.open(file_path).convert('RGB')
		image_vector = np.asarray(image)
		
	except Exception:
		image_vector = None
	
	return image_vector
	
	
"""
	Изменение размеров холста картинки
"""
def image_resize_canvas(image, width, height):
	
	pixels = image.load()
	
	image_new = Image.new('RGB', (width, height), color = pixels[0, 0])
	draw = ImageDraw.Draw(image_new)
	
	position = (
		math.ceil((width - image.size[0]) / 2),
		math.ceil((height - image.size[1]) / 2),
	)
	
	image_new.paste(image, position)
	
	del pixels, draw, image
	
	return image_new


class DataSet:
	
	
	def __init__(self):
		self.file_path = None
		self.file_path_tmp = None
		
		self.zip_file = None
		self.zip_file_namelist = []
		
		self.zip_file_tmp = None
		self.zip_file_tmp_namelist = []
	
	
	"""
		Открывает zip Архив датасета
	"""
	def open(self, file_path):
		self.file_path = file_path
		self.file_path_tmp = file_path + ".tmp"
		self.zip_file = zipfile.ZipFile(file_path, 'a')
		self.zip_file_namelist = self.zip_file.namelist()
	
	
	"""
		Открывает временный zip архив датасета для записи
	"""
	def open_tpm_write(self):
		if self.zip_file_tmp is None:
			self.zip_file_tmp = zipfile.ZipFile(self.file_path_tmp, 'w')
			self.zip_file_tmp_namelist = []
	
	
	"""
		Пересоздает оригинальный zip файл с изменениями
	"""
	def flush(self):
		
		if self.zip_file_tmp is not None:
			for file_name in self.zip_file_namelist:
				file_index = indexOf(self.zip_file_tmp_namelist, file_name)
				if file_index == -1:
					data = self.zip_file.read(file_name)
					self.zip_file_tmp.writestr(file_name, data)
					self.zip_file_tmp_namelist.append(file_name)
						
			self.zip_file.close()
			self.zip_file_tmp.close()
			
			shutil.move(self.file_path_tmp, self.file_path)
			
			if self.file_path_tmp != "" and os.path.isfile(self.file_path_tmp):
				os.unlink(self.file_path_tmp)
			
			self.open(self.file_path)
	
	
	"""
		Завершает работу с архивом
	"""
	def close(self):
		self.flush()
		
		if self.zip_file_tmp is not None:
			self.zip_file.close()
		
		if self.zip_file_tmp is not None:
			self.zip_file_tmp.close()
		
		self.zip_file = None
		self.zip_file_tmp = None
		self.zip_file_namelist = []
		self.zip_file_tmp_namelist = []
		
		if self.file_path_tmp != "" and os.path.isfile(self.file_path_tmp):
			os.unlink(self.file_path_tmp)
		
		self.file_path = ""
		self.file_path_tmp = ""
	
	
	"""
		Сохраняет поток байтов в zip архив
	"""
	def write_bytes(self, file_name, data):
		
		index = indexOf(self.zip_file_namelist, file_name)
		
		if index == -1:
			self.zip_file.writestr(file_name, data)
			self.zip_file_namelist.append(file_name)
			
		else:
			
			index = indexOf(self.zip_file_tmp_namelist, file_name)
			if index != -1:
				self.flush()
				
			self.open_tpm_write()
			self.zip_file_tmp.writestr(file_name, data)
			self.zip_file_tmp_namelist.append(file_name)
		
		pass
	
	
	"""
		Сохраняет данные в zip архив
	"""
	def save_file(self, file_name, data):
		
		bytes = None
		
		if isinstance(data, Image.Image):
			tmp = io.BytesIO()
			data.save(tmp, format='PNG')
			bytes = tmp.getvalue()
		
		if (isinstance(data, str)):
			bytes = data.encode("utf-8")
		
		if bytes is not None:
			self.write_bytes(file_name, bytes)
		
		pass
	
	
	"""
		Возвращает путь к картинке
	"""
	def get_captcha_path(self, photo_number):
		
		parent_image_dir = str(photo_number % 100)
		parent_image_dir = parent_image_dir.zfill(2)
		file_name = str(photo_number)
		
		image_result_path = os.path.join("captcha", parent_image_dir, file_name) + ".png"
		image_mask_path = os.path.join("captcha", parent_image_dir, file_name) + "-mask.png"
		image_json_path = os.path.join("captcha", parent_image_dir, file_name) + "-json.txt"
		
		return {
			"dir": os.path.join("captcha", parent_image_dir),
			"image": image_result_path,
			"mask": image_mask_path,
			"json": image_json_path,
		}
	
	
	"""
		Загружает картинку капчи из zip файла
	"""
	def get_captcha(self, photo_number):
		
		captcha = None
		path = self.get_captcha_path(photo_number)
		
		try:
			captcha = Captcha()
			captcha.image = self.zip_file.read(path["image"])
			captcha.mask = self.zip_file.read(path["mask"])
			image_json = self.zip_file.read(path["json"])
			image_json = image_json.decode("utf-8")
			captcha.load_json(image_json)
			
			if not captcha.is_load():
				captcha = None
			
		except Exception:
			captcha = None
		
		return captcha
	
	
	"""
		Сохраняет картинку капчи в zip файл
	"""
	def save_captcha(self, photo_number, captcha):
		
		path = self.get_captcha_path(photo_number)
		
		self.save_file(path["image"], captcha.image)
		self.save_file(path["mask"], captcha.mask)
		self.save_file(path["json"], captcha.get_json())
		
	
	"""
		Генерация капчи
	"""
	def generate_captcha(self, photo_number, force=False):
		
		# Проверка есть ли такой уже файл
		if not force:
			captcha = self.get_captcha(photo_number)
			if captcha is not None:
				return
		
		captcha = Captcha()
		captcha.generate()
		captcha.resize_max()
		
		self.save_captcha(photo_number, captcha)
		
		pass
	
	
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
		
		image_mask = self.mask.copy()
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
		
		fig = plt.figure()
		
		fig.add_subplot(rows, columns, 1)
		plt.imshow(self.image, cmap='gray')
		plt.title(
			"Size: " +
			str(self.image.size[0]) + "x" + str(self.image.size[1]) +
			". Answer: " + self.answer.upper())
		
		fig.add_subplot(rows, columns, 2)
		plt.imshow(image_mask, cmap='gray')
		
		plt.savefig("tmp/captcha.png")
		plt.show()
	
	
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
	
	
	def create_image(self, color):
		img = Image.new('RGB', (self.image_width, self.image_height), color = color)
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
		image_result, draw_result = self.create_image( self.image_color )
		image_mask, draw_mask = self.create_image("white")
		
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

