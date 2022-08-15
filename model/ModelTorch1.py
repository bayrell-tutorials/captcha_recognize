# -*- coding: utf-8 -*-

##
# Copyright (с) Ildar Bikmamatov 2022
# License: MIT
##

import os, torch

from torch.nn import Module
from typing import Optional
from torch import nn
from ai_helper import *



class NetworkTorch1(TorchNetwork):
	
	
	def __init__(self):
		
		AbstractNetwork.__init__(self)
		
		self.batch_size = 64
		self.epochs = 30
		self.model_name = os.path.join("data", "model", "torch1")
	
	
	
	def create_model(self):
		
		"""
		Создает модель
		"""
		
		TorchNetwork.create_model(self)
		
		self.model = DirectModule()
		
		# Входной слой
		self.model.add_module( "input", nn.Linear(self.input_shape, 128) )
		self.model.add_module( "input_relu", nn.ReLU(), "input" )
		
		# Выходной слой
		self.model.add_module( "output",
			nn.Linear(
				self.model["input"].out_features,
				self.output_shape
			),
			"input_relu"
		)
		self.model.add_module( "output_softmax", nn.Softmax(), "output" )
		
		# Устанавливаем выходной слой
		self.model.set_output_module("output_softmax")
		
		# Adam optimizer
		self.optimizer = torch.optim.Adam( self.model.parameters() )
		
		# mean squared error
		self.loss = nn.MSELoss()
		
		pass