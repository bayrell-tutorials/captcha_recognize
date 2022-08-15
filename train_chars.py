#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

##
# Copyright (с) Ildar Bikmamatov 2022
# License: MIT
##

import os, sys, random, math

#os.environ["TF_GPU_ALLOCATOR"]="cuda_malloc_async"
os.environ["TF_CPP_VMODULE"]="gpu_process_state=10,gpu_cudamallocasync_allocator=10"

from ai_helper import *
from model.lib import DATASET_CHARS, DATASET_CHARS_COUNT, get_model
from dataset import create_random_chars_dataset, load_train_dataset_chars


# Инициализация
random.seed()

#sys.exit()


def do_create(model: AbstractModel):
	
	"""
		Создаем модель
	"""
	
	model.load()
	
	# Если модель не загружена, то создаем ее
	if not model.is_loaded():

		# Создаем модель
		model.create()
		model.show_summary()


def do_train(model: AbstractModel):
	
	"""
		Обучаем модель
	"""
	
	# Если модель не загружена, то создаем и обучаем ее
	if model.is_new():
		
		# Загружаем обучающий датасет
		train_dataset = load_train_dataset_chars()
		train_dataset.build_train()
		
		input_shape, output_shape, train_count, test_count = train_dataset.get_train_shape()
		
		print ("=========================")
		print ("Dataset info:")
		print ("Input shape:", input_shape)
		print ("Output shape:", output_shape)
		print ("Train count:", train_count)
		print ("Test count:", test_count)
		print ("=========================")
		
		# Обучаем модель
		model.set_dataset(train_dataset)
		model.train()
		model.show_train_info()
		
	else:
		
		#model.show()
		pass



def check_answer(question, answer, control):
	
	"""
		Проверка ответа
	"""
	
	answer_value = get_answer_from_vector(answer)
	control_value = get_answer_from_vector(control)
	
	if answer_value != control_value:
		title = DATASET_CHARS[answer_value] + " | " + DATASET_CHARS[control_value]
		#print (title)
	
	return answer_value == control_value



def check_model(model: AbstractModel):
	
	"""
		Проверка модели
	"""
	
	control_dataset = create_random_chars_dataset(1000)
	correct_answers, total_questions = model.check(
		control_dataset=control_dataset,
		callback=check_answer
	)
	
	rate = math.ceil(correct_answers / total_questions * 100)
	print ("Correct answers: " + str(correct_answers) + " of " + str(total_questions))
	print ("Rate: " + str(rate) + "%")



if __name__ == '__main__':
	
	# Настройка GPU
	# tensorflow_gpu_init(1024)
	
	# Запуск
	#model = get_model("data/model/keras1")
	model = get_model("data/model/torch1")
	
	model.input_shape = (32, 32)
	model.output_shape = (DATASET_CHARS_COUNT)
	
	print ("")
	print ("")
	print ("=====================================")
	print ("Train model: ", model.get_model_name())
	print ("")
	print ("")
	
	do_create(model)
	do_train(model)
	
	print ("")
	print ("")
	print ("=====================================")
	print ("Check model: ", model.get_model_name())
	check_model(model)

	pass
