#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

##
# Copyright (с) Ildar Bikmamatov 2022
# License: MIT
##

import os, sys, random, math

#os.environ["TF_GPU_ALLOCATOR"]="cuda_malloc_async"
os.environ["TF_CPP_VMODULE"]="gpu_process_state=10,gpu_cudamallocasync_allocator=10"

from model.Model_1 import Model_1
from model.Model_2 import Model_2
from lib.Helper import tensorflow_gpu_init
from lib.DataSet import get_answer_from_vector, \
	get_train_dataset_chars, get_train_dataset_chars2, \
	DATASET_CHARS


# Инициализация
random.seed()
tensorflow_gpu_init(1024)

#sys.exit()


def do_train(model):
	
	# Загружаем модель
	model.load()
	
	# Если модель не загружена, то создаем и обучаем ее
	if not model.is_loaded():
		
		# Загружаем обучающий датасет
		train_dataset = get_train_dataset_chars()
		train_dataset.build()
		model.set_dataset(train_dataset)
		
		input_shape, output_shape, train_count, test_count = train_dataset.get_build_shape()
		
		print ("=========================")
		print ("Dataset info:")
		print ("Input shape:", input_shape)
		print ("Output shape:", output_shape)
		print ("Train count:", train_count)
		print ("Test count:", test_count)
		print ("=========================")
		
		# Создаем модель
		model.create()
		model.show()
		
		# Обучаем модель
		model.train()
		model.train_show()
		
	else:
		
		model.show()
		pass


# Проверка ответа
def check_answer(question, answer, control):
	
	answer_value = get_answer_from_vector(answer)
	control_value = get_answer_from_vector(control)
	
	if answer_value != control_value:
		title = DATASET_CHARS[answer_value] + " | " + DATASET_CHARS[control_value]
		#print (title)
	
	return answer_value == control_value


def check_model(model):
	
	control_dataset = get_train_dataset_chars2(1000)
	correct_answers, total_questions = model.check(
		control_dataset=control_dataset,
		callback=check_answer
	)
	
	rate = math.ceil(correct_answers / total_questions * 100)
	print ("Correct answers: " + str(correct_answers) + " of " + str(total_questions))
	print ("Rate: " + str(rate) + "%")


# Запуск
model = Model_1()

print ("Model: ", model.get_model_name())
do_train(model)
check_model(model)

