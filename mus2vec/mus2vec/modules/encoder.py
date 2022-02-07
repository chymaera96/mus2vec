import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional


class Encoder(nn.Module):
	def __init__(self, in_channels=1, kernel_size=(5,3), padding=(2,1)):
		super(Encoder, self).__init__()
		self.in_channels = in_channels
		self.kernel_size = kernel_size
		self.padding = padding
		self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=kernel_size, padding=padding)
		self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=kernel_size, padding=padding)
		self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=kernel_size, padding=padding)
		self.pool1 = nn.MaxPool2d((2,4), stride=(2,4))
		self.pool2 = nn.MaxPool2d((3,4), stride=(3,4))
		self.pool3 = nn.MaxPool2d((2,4), stride=(2,4))
		self.relu = nn.ReLU()
		
	def forward(self, x):
		x = self.conv1(x)
		x = self.relu(x)
		x = self.pool1(x)

		x = self.conv2(x)
		x = self.relu(x)
		x = self.pool2(x)

		x = self.conv3(x)
		x = self.relu(x)
		x = self.pool3(x)

		x = x.reshape(x.shape[0],-1)
		return x
		
	