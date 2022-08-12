#!/usr/bin/env python3

from genericpath import isdir
import random, math, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

random.seed()

def sign(x):
	if x >= 0: return 1
	return -1


def generate_captcha():
	
	text_str="1234567890QWERTYUIOPASDFGHJKLZXCVBNMqwertyuiopasdfghjklzxcvbnm"
	text_str_count = len(text_str)
	image_width = 300
	image_height = 90
	img_center = (image_width/2, image_height/2)
	image_result = None
	draw_result = None
	image_mask = None
	draw_mask = None
	dots_count = 10
	lines_count = 2
	text_count = 6
	
	def get_random_font():
		
		font_arr = [
			"fonts/Roboto/Roboto-Bold.ttf",
			"fonts/Roboto/Roboto-Regular.ttf",
			"fonts/Roboto/Roboto-Italic.ttf",
			"fonts/Montserrat/static/Montserrat-Bold.ttf",
			"fonts/Montserrat/static/Montserrat-Italic.ttf",
			"fonts/Montserrat/static/Montserrat-Regular.ttf",
		]
		
		font_path = font_arr[ random.randint(0, len(font_arr) - 1) ]
		font = ImageFont.truetype(font_path, random.randint(28, 32))
		
		return font
	
	
	def create_image(color):
		img = Image.new('RGB', (image_width, image_height), color = color)
		draw = ImageDraw.Draw(img)
		return (img, draw)
	
	
	def draw_line(color, width):
		
		padding_x = 25
		padding_y = 10
		
		position = (
			random.randint(padding_x, padding_x),
			random.randint(padding_y, image_height - padding_y),
			random.randint(image_width - padding_x, image_width - padding_x),
			random.randint(padding_y, image_height - padding_y)
		)
		
		draw_result.line(position, fill=color, width=width)
		
		pass
	
	
	def draw_dots(color, width):
		
		padding_x = 25
		padding_y = 10
		x = random.randint(padding_x, image_width - padding_x)
		y = random.randint(padding_y, image_height - padding_y)
		
		position = (x,y,x+width,y+width)
		draw_result.ellipse(position, fill=color, width=width)
	

	def draw_text(position, text, color, font, angle):
		
		text_box_size = font.getsize(text)
		if text_box_size[0] > text_box_size[1]:
			text_box_size = (text_box_size[0], text_box_size[0])
		else:
			text_box_size = (text_box_size[1], text_box_size[1])
		
		img = Image.new('L', text_box_size, color="black")
		draw = ImageDraw.Draw(img)
		draw.text((0,0), text, "white", font=font)
		
		img = img.rotate(angle,  expand=1)
		
		position = (
			math.ceil(position[0] - text_box_size[0] / 2),
			math.ceil(position[1] - text_box_size[1] / 2)
		)
		
		image_result.paste(ImageOps.colorize(img, black=(0, 0, 0), white=color), position, img)
		image_mask.paste(ImageOps.colorize(img, black=(0, 0, 0), white="black"), position, img)
		
		del draw
		del img
		
		return text_box_size
		
	
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
	
	color = colors[ random.randint(0, len(colors) - 1) ]
	image_result, draw_result = create_image( color )
	image_mask, draw_mask = create_image("white")
	
	image_answer = ""
	step_text = image_width / text_count
	step_text_x = math.ceil(step_text / 4)
	step_text_y = math.ceil(image_height / 5)
	for i in range(0, text_count):
		
		pos_x = step_text * i + step_text / 2
		pos_y = image_height / 2
		
		pos_x += random.randint(-step_text_x, step_text_x)
		pos_y += random.randint(-step_text_y, step_text_y)
		
		angle = random.randint(-50, 50)
		text = text_str[ random.randint(0, text_str_count - 1) ]
		image_answer += text
		font = get_random_font()
		
		color = (
			random.randint(10, 200),
			random.randint(10, 200),
			random.randint(10, 200),
		)
		
		draw_text(
			position=(pos_x, pos_y),
			text=text,
			color=color,
			font=font,
			angle=angle
		)
	
		del font
	
	
	for i in range(0, lines_count):
		color = (
			random.randint(10, 200),
			random.randint(10, 200),
			random.randint(10, 200),
		)
		draw_line(
			color=color,
			width=random.randint(2, 3) 
		)
	
	
	for i in range(0, dots_count):
		color = (
			random.randint(10, 200),
			random.randint(10, 200),
			random.randint(10, 200),
		)
		draw_dots(
			color=color,
			width=random.randint(3, 5) 
		)
	
	del draw_result
	del draw_mask
	
	return (image_result, image_mask, image_answer)
	
	
def generate_test_image():	
	image_result, image_mask, image_answer = generate_captcha()
	
	print(image_answer)
	#image_result.show()
	image_mask.show()


def generate_images():
	
	if not os.path.isdir("images"):
		os.makedirs("images")

	count_images = 1000
	for i in range(0, count_images):
		
		parent_image_dir = str(i % 100)
		parent_image_dir = parent_image_dir.zfill(2)
		file_name = str(i).zfill(4)
		
		parent_image_dir_path = os.path.join("images", parent_image_dir)
		if not os.path.isdir(parent_image_dir_path):
			os.makedirs( parent_image_dir_path )
		
		image_result_path = os.path.join("images", parent_image_dir, file_name) + ".jpg"
		image_mask_path = os.path.join("images", parent_image_dir, file_name) + "-mask.jpg"
		image_answer_path = os.path.join("images", parent_image_dir, file_name) + "-answer.txt"
		
		image_result, image_mask, image_answer = generate_captcha()
		
		print(file_name + ".jpg")
		image_result.save(image_result_path)
		image_mask.save(image_mask_path)
		
		with open(image_answer_path, 'w', encoding='utf-8') as file:
			file.write(image_answer)
		
		
def convert_image_to_vector(file_path):
	
	if not os.path.isfile(file_path):
		return None
	
	vector = None
	
	try:
		image = Image.open(file_path).convert('RGB')
		vector = np.asarray(image)
		
	except Exception:
		vector = None
	
	return vector
		
		
def load_image(parent_image_dir, file_name):
	
	image_result_path = os.path.join("images", parent_image_dir, file_name) + ".jpg"
	image_mask_path = os.path.join("images", parent_image_dir, file_name) + "-mask.jpg"
	image_answer_path = os.path.join("images", parent_image_dir, file_name) + "-answer.txt"
	
	image_answer = ""
	image_result_vector = convert_image_to_vector(image_result_path)
	image_mask_vector = convert_image_to_vector(image_mask_path)
	
	try:
		with open(image_answer_path, 'r', encoding='utf-8') as file:
			image_answer = file.readline()
	except Exception:
		image_answer = ""
	
	if image_result_vector is None or image_mask_vector is None or image_answer == "":
		return None
	
	return np.array([
		image_result_vector,
		image_mask_vector,
		image_answer
		#np.array(list(image_answer.encode('utf8')))
	], dtype=object)


def load_dataset():
	
	res = None
	
	count_images = 2
	for i in range(0, count_images):
	
		parent_image_dir = str(i % 100)
		parent_image_dir = parent_image_dir.zfill(2)
		file_name = str(i).zfill(4)	
		
		data = load_image(parent_image_dir, file_name)
		if data is None:
			continue
		
		if res is None:
			res = np.expand_dims(data, axis=0)
		else:
			res = np.append(res, [data], axis=0)
		
	return res


def save_dataset():
	dataset = load_dataset()
	np.save("captcha_dataset", dataset, allow_pickle=True)


generate_images()
save_dataset()

print ("Ok")