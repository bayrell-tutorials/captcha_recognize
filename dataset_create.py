#!/usr/bin/env python3
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



def generate_captcha_and_show():
	captcha = Captcha()

	color = "black"
	font = captcha.get_font(number=1, size=32)
	angle=-45
	#angle=0

	#image = captcha.get_rotated_text("W", font, angle)
	#plt.imshow(image, cmap='gray')
	#plt.show()

	captcha.generate()
	captcha.resize_max()

	captcha.show()



def dataset_test():
	
	dataset = DataSet()
	dataset.open("data/captcha_dataset.zip")
	
	captcha = dataset.get_captcha(1)
	print (captcha)
	
	dataset.close();


generate_captcha_dataset()
#generate_captcha_and_show()
#dataset_test()
