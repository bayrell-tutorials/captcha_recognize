# -*- coding: utf-8 -*-

##
# Copyright (с) Ildar Bikmamatov 2022
# License: MIT
##

import io, os, random, math
import numpy as np

from PIL import Image
from sklearn.model_selection import train_test_split
from .Helper import vector_append
from .Image import image_to_vector
from .Captcha import Captcha, generate_captcha_char


# Constant
DATASET_CHARS = "1234567890QWERTYUIOPASDFGHJKLZXCVBNM"
DATASET_CHARS_COUNT = len(DATASET_CHARS)
DATASET_DIR = "data"


"""
	Кодирование символа
"""
def dataset_number_to_symbol(char_number):
	return DATASET_CHARS[ char_number % DATASET_CHARS_COUNT ]


"""
	Возвращает выходной вектор
"""
def get_output_vector_by_number(number, count):
	res = [0.0] * count
	if (number >=0 and number < count):
		res[number] = 1.0
	return np.asarray(res)


"""
	Возвращает ответ. Позиция максимального значения в векторе будет ответом
"""
def get_answer_from_vector(vector):
	value_max = -math.inf
	value_index = 0
	for i in range(0, len(vector)):
		value = vector[i]
		if value_max < value:
			value_index = i
			value_max = value
	
	return value_index


"""
	Возвращает путь к картинке
"""
def get_captcha_path(photo_number):
	
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



"""
	Возвращает обучающий вектор по изображению для символов
"""
def get_train_vector_chars(char_number, image):
	
	question_vector = image_to_vector(image)
	question_vector = question_vector.astype('float32') / 255.0
	
	answer_vector = get_output_vector_by_number(char_number, DATASET_CHARS_COUNT)
	
	return question_vector, answer_vector
	
	
	
"""
	Вовзращает датасет chars
"""
def get_train_dataset_chars():
	
	dataset = DataSet()
	
	dataset_reader = DataSetReader()
	dataset_reader.open("data_chars")
	
	for char_number in range(0, DATASET_CHARS_COUNT):
		
		char = DATASET_CHARS[ char_number ]
		files = dataset_reader.files(char)
		
		for file in files:
			
			# Получаем изображение
			image = dataset_reader.read_file(file)
			
			x, y = get_train_vector_chars(char_number, image)
			dataset.append(x, y)
	
	return dataset



"""
	Генерация рандомного датасета
"""
def get_train_dataset_chars2(count=1000):
	
	dataset = DataSet()
	
	for i in range(0, count):
		
		char_number = random.randint(0, DATASET_CHARS_COUNT - 1)
		char = DATASET_CHARS[ char_number ]
		
		angle = random.randint(-50, 50)
		font_size = random.randint(28, 36)
		
		# Генерация изображения
		image = generate_captcha_char(
			char,
			size=font_size,
			angle=angle
		)
		
		x, y = get_train_vector_chars(char_number, image)
		dataset.append(x, y)
	
	
	return dataset



class DataSet:
	
	def __init__(self):
		
		self.data = []
		self.train_x = None
		self.train_y = None
		self.test_x = None
		self.test_y = None
		self._is_new = False
	
	
	"""
		Добавить в датасет данные
	"""
	def append(self, x, y):
		
		self.data.append((x, y))
		self._is_new = True
	
	
	def get_x(self):
		return np.asarray(list(map(lambda item: item[0], self.data)))
	
	
	def get_y(self):
		return np.asarray(list(map(lambda item: item[1], self.data)))
	
	
	"""
		Собрать датасет
	"""
	def build(self):
		
		if self._is_new and self.data != None:
			
			train, test = train_test_split(self.data)
			
			self.train_x = np.asarray(list(map(lambda item: item[0], train)))
			self.train_y = np.asarray(list(map(lambda item: item[1], train)))
			
			self.test_x = np.asarray(list(map(lambda item: item[0], test)))
			self.test_y = np.asarray(list(map(lambda item: item[1], test)))
			
			self._is_new = False
			
			del train, test
	
	
	"""
		Возвращает размеры входного векторов
	"""
	def get_input_shape(self):
		input_shape = self.data[0][0].shape
		return input_shape
		
		
	"""
		Возвращает размеры выходного векторов
	"""
	def get_output_shape(self):
		output_shape = self.data[0][1].shape
		return output_shape
		
		
	"""
		Возвращает размеры входного и выходного векторов
	"""
	def get_build_shape(self):
		
		input_shape = self.train_x.shape[1:]
		output_shape = self.train_y.shape[1]
		train_count = self.train_x.shape[0]
		test_count = self.test_x.shape[0]
		
		return input_shape, output_shape, train_count, test_count
	
	
	
class DataSetReader:
	
	def __init__(self):
		self.dataset_name = "dataset"
	
	
	"""
		Открыть датасет
	"""
	def open(self, dataset_name):
		self.dataset_name = dataset_name
	
	
	"""
		Сбрасывает изменения на диск
	"""
	def flush(self):
		pass
	
	
	"""
		Завершает работу с датасетом
	"""
	def close(self):
		self.flush()
	
	
	def get_dataset_path(self, file_name = ""):
		if file_name == "":
			return os.path.join(DATASET_DIR, self.dataset_name)
		return os.path.join(DATASET_DIR, self.dataset_name, file_name)
	
	
	"""
		Возвращает список файлов в папке
	"""
	def files(self, path):
	
		def read_dir(path):
			res = []
			items = os.listdir(path)
			for item in items:
				
				item_path = os.path.join(path, item)
				
				if item_path == "." or item_path == "..":
					continue
				
				if os.path.isdir(item_path):
					res = res + read_dir(item_path)
				else:
					res.append(item_path)
				
			return res
		
		dir_name = self.get_dataset_path()
		items = read_dir(  os.path.join(dir_name, path) )
		
		def f(item):
			return item[len(dir_name + "/"):]
		
		items = list( map(f, items) )
		
		return items
	
	
	"""
		Сохраняет байты в файл
	"""
	def save_bytes(self, file_name, data):
		
		file_path = self.get_dataset_path(file_name)
		file_dir = os.path.dirname(file_path)
		
		if not os.path.isdir(file_dir):
			os.makedirs(file_dir)
		
		f = open(file_path, 'wb')
		f.write(data)
		f.close()
		
	
	"""
		Читает из файла байты
	"""
	def read_bytes(self, file_name):
		
		file_path = self.get_dataset_path(file_name)
		
		f = open(file_path, 'rb')
		data = f.read()
		f.close()
		
		return data
	
		
	"""
		Сохраняет данные в файл
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
			self.save_bytes(file_name, bytes)
		
		pass
	
	
	
	"""
		Читает данные из файла
	"""
	def read_file(self, file_name):
		return self.read_bytes(file_name)
	
	
	
	"""
		Загружает картинку капчи из файла
	"""
	def get_captcha(self, photo_number):
		
		captcha = None
		path = get_captcha_path(photo_number)
		
		try:
			captcha = Captcha()
			captcha.image = self.read_file(path["image"])
			captcha.mask = self.read_file(path["mask"])
			image_json = self.read_file(path["json"])
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
		
		self.save_file(path["image"], image)
		self.save_file(path["mask"], mask)
		self.save_file(path["json"], captcha.get_json())
		
		del image, mask
		
		
		
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
	
	
	