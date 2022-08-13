#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

##
# Copyright (с) Ildar Bikmamatov 2022
# License: MIT
##

import math, random
import tensorflow.keras as keras

from lib import *
from sklearn.model_selection import train_test_split

model_name = "model_chars"
train_number = 3
dataset_path = "data/captcha_dataset.zip"
model_train_name = model_name + "_" + str(train_number)

print ( model_train_name )

random.seed()

"""
	Датасет из zip файла
"""
def get_train_dataset():
	
	res_question = None
	res_answer = None

	dataset = DataSet()
	dataset.open(dataset_path)
	
	for char_number in range(0, DATASET_CHARS_COUNT):
		
		char = DATASET_CHARS[ char_number ]
		files = dataset.files("chars/" + char)
		for file in files:
			
			# Получаем изображение
			image = dataset.read_file(file)
			
			# Получаем вектора
			question_vector, answer_vector = get_train_vector_chars(char_number, image)
			
			# Добавляем вектора в результат
			res_question = vector_append(res_question, question_vector)
			res_answer = vector_append(res_answer, answer_vector)
			
	return (res_question, res_answer)


"""
	Рандомная обучающий датасет
"""
def get_train_dataset2(count=1000):
	
	res_question = None
	res_answer = None
	
	for i in range(0, count):
		
		char_number = random.randint(0, DATASET_CHARS_COUNT - 1)
		char = DATASET_CHARS[ char_number ]
		
		#angle = random.randint(-50, 50)
		angle = 0
		font_size = random.randint(28, 36)
		
		# Генерация изображения
		image = generate_captcha_char(
			char,
			size=font_size,
			angle=angle
		)
		
		# Получаем вектора
		question_vector, answer_vector = get_train_vector_chars(char_number, image)
		
		# Добавляем вектора в результат
		res_question = vector_append(res_question, question_vector)
		res_answer = vector_append(res_answer, answer_vector)
		
	return res_question, res_answer
	

"""
	Создание модели
"""
def create_model(input_shape, output_shape, train_number):
	
	from tensorflow.keras.models import Sequential
	from tensorflow.keras.layers import Dense, Input, Flatten, \
		Dropout, Conv2D, MaxPooling2D, Reshape
	
	model_train_name = model_name + "_" + str(train_number)
	model = Sequential(name=model_train_name)
	
	input_shape = input_shape[1:]
	output_shape = output_shape[1]
	
	# Входной слой
	model.add(Input(input_shape, name='input'))
	
	# Reshape
	model.add(Reshape( target_shape=(input_shape[0], input_shape[1], 1) ))
	
	# Сверточный слой
	model.add(Conv2D(128, kernel_size=(3, 3), padding="same", activation="relu"))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	
	# Сверточный слой
	model.add(Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	
	# Сверточный слой
	#model.add(Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu"))
	#model.add(MaxPooling2D(pool_size=(2, 2)))
	
	# Выравнивающий слой
	model.add(Flatten())
	
	# Полносвязные слои
	#model.add(Dense(256, activation='relu'))
	#model.add(Dropout(0.5))
	
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.5))
	
	# Выходной слой
	model.add(Dense(output_shape, name='output', activation='softmax'))
	
	# Среднеквадратическая функция ошибки
	model.compile(
		loss='mean_squared_error', 
		optimizer='adam',
		metrics=['accuracy'])
	
	# Вывод на экран информация о модели
	model.summary()
	
	file_name = "data/" + model_train_name + "_plot.png"
	
	keras.utils.plot_model(
		model,
		to_file=file_name,
		show_shapes=True)
	
	#img = pltimg.imread(file_name)
	#plt.imshow(img)
	#plt.show()
	
	return model
	
	
"""
	Обучение модели
"""
def train_model(model, train_x, train_y, test_x, test_y, train_number=1):
	
	checkpoint_path = "data/"  + model_name + "/training_" + str(train_number) + "/cp.ckpt"
	checkpoint_dir = os.path.dirname(checkpoint_path)
	
	# Создаем папку, куда будут сохраняться веса во время обучения
	if not os.path.isdir(checkpoint_dir):
		os.makedirs(checkpoint_dir)
	
	# Callback функия для сохранения весов
	cp_callback = keras.callbacks.ModelCheckpoint(
		filepath=checkpoint_path,
		save_weights_only=True,
		verbose=1
	)
	
	history = model.fit(
		
		# Входные данные
		train_x,
		
		# Выходные данные
		train_y,
		
		# Размер партии для обучения
		batch_size=512,
		
		# Количество эпох обучения
		epochs=20,
		
		# Контрольные данные
		validation_data=(test_x, test_y),
		
		# Подробный вывод
		verbose=1,
		
		# Сохраняем контрольные точки
		callbacks=[cp_callback]
	) 
	
	# Сохраняем модель на диск
	model.save('data/' + model_name)
	model.save('data/' + model_name + ".h5")
	model.save('data/' + model_train_name + ".h5")
	
	total_val_accuracy = math.ceil(history.history['val_accuracy'][-1] * 100)
	
	# Сохраняем картинку
	plt.title("Итог: " + str(total_val_accuracy) + "%")
	plt.plot( np.multiply(history.history['accuracy'], 100), label='Обучение')
	plt.plot( np.multiply(history.history['val_accuracy'], 100), label='Контрольные ответы')
	plt.plot( np.multiply(history.history['loss'], 100), label='Ошибка')
	plt.ylabel('Процент')
	plt.xlabel('Эпоха')
	plt.legend()
	plt.savefig('data/' + model_train_name + '_history.png')
	plt.show()
	
	return history
	

def do_train(train_number = 1):
	
	res = get_train_dataset()
	
	print ("Shape question:", res[0].shape)
	print ("Shape answer:", res[1].shape)
	
	model = create_model(res[0].shape, res[1].shape, train_number)

	train_x, test_x, train_y, test_y = train_test_split(res[0], res[1])

	print("Train", train_x.shape, "=>", train_y.shape)
	print("Test", test_x.shape, "=>", test_y.shape)

	train_model(model, train_x, train_y, test_x, test_y, train_number)


def show_train_model(train_number):
	
	model = keras.models.load_model('data/' + model_train_name + '.h5')
	
	# Вывод на экран информация о модели
	model.summary()
	
	file_name = "data/" + model_train_name + "_plot.png"
	
	keras.utils.plot_model(
		model,
		to_file=file_name,
		show_shapes=True
	)
	
	
def do_check(count=10000):
	
	dataset = DataSet()
	dataset.open(dataset_path)
	
	#model = keras.models.load_model('data/' + model_name)
	model = keras.models.load_model('data/' + model_train_name + '.h5')
	
	res_question = None
	
	def create_question_random_chars(count):
		
		res_question = None
		res_control = []
		
		for i in range(0, count):
			
			char_number = random.randint(0, DATASET_CHARS_COUNT - 1)
			char = DATASET_CHARS[ char_number ]
			angle = random.randint(-50, 50)
			#angle = 0
			font_size = random.randint(28, 36)
			
			image = generate_captcha_char(
				char,
				size=font_size,
				angle=angle
			)
			
			# Получаем вектора
			question_vector, _ = get_train_vector_chars(char_number, image)
			
			# Добавляем вектора в результат
			res_question = vector_append(res_question, question_vector)
			res_control.append(char_number)
			
		return res_question, res_control
		
	
	# Формируем запрос
	res_question, res_control = create_question_random_chars(count)
	#res_question, res_control = create_question_from_dataset(dataset)
	
	# Спрашиваем модель
	res_answer = model.predict( res_question )
	
	# Выводим ответы
	correct_answers = 0
	res_answer_count = len(res_answer)
	
	for i in range(0, res_answer_count):
		
		control_value = res_control[i]
		answer_vector = res_answer[i]
		answer_value = get_answer_from_vector(answer_vector)
		
		if answer_value != control_value:
			
			title = DATASET_CHARS[answer_value] + " | " + DATASET_CHARS[control_value]
			print (title)
		
			image = res_question[i]
			
			#plt.title(title)
			#plt.imshow(image, cmap='gray')
			#plt.show()
			
		
		if control_value == answer_value:
			correct_answers = correct_answers + 1
		pass
	
	rate = math.ceil(correct_answers / res_answer_count * 100)
	print ("Correct answers: " + str(correct_answers) + " of " + str(res_answer_count))
	print ("Rate: " + str(rate) + "%")
	
	pass


#do_train(train_number)
do_check(1000)

#show_train_model(1)
