# -*- coding: utf-8 -*-

##
# Copyright (с) Ildar Bikmamatov 2022
# License: MIT
##

import os, zipfile, shutil
from .Helper import indexOf


class DataSetZipReader:
	
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
		Список файлов
	"""
	def files(self, file_name = ""):
		def f(name):
			return name.find(file_name) == 0
		return list(filter(f, self.zip_file_namelist[:]))
	
	
	"""
		Сохраняет поток байтов в zip архив
	"""
	def save_bytes(self, file_name, data):
		
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
		Читает из файла байты
	"""
	def read_bytes(self, file_name):
		return self.zip_file.read(file_name)
		