import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import time
import scipy.io
import h5py
import hdf5storage
from numpy import random
import math

from random import shuffle

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import os
import scipy
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import platform

from argparse import ArgumentParser
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torchvision.models import AlexNet
import argparse
from scipy.io import loadmat
import scipy.io as sio

from scipy.ndimage.filters import maximum_filter, gaussian_filter, median_filter,uniform_filter

parser = ArgumentParser(description='CMFNet')
parser.add_argument('--imgname', type=str, default='scene3', help='captured image name')
args = parser.parse_args()


num_x = 256
num_y = 256
size_img = 256
num_filter = 8
num_wavelength = 31
layer_num = 5
start_epoch = 7
end_epoch = 100
learning_rate = 1e-3
log_file_name = 'output_f3.txt'

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImageUnshuffle(nn.Module):
	def __init__(self, scale=4):
		super().__init__()
		self.scale = scale

	def forward(self, x):
		N, C, H, W = x.size()
		S = self.scale
		x = x.view(N, C, H // S, S, W // S, S)  # (N, C, H//bs, bs, W//bs, bs)
		x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
		x = x.view(N, S * S * C, H // S, W // S) 
		return x


class ImageShuffle(nn.Module):
	def __init__(self, scale=4):
		super().__init__()
		self.scale = scale
		
	def forward(self, x):
		N, S2C, H, W = x.size()
		S = self.scale
		C = int(S2C/S/S)
#		print(N, S, S, C, H, W)
		x = x.view(N, S, S, C, H, W)
		x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
		x = x.view(N, C, H * S, W * S)  

		return x
		
		

class NonLocalBlock2D(nn.Module):
	def __init__(self, scale):
		super(NonLocalBlock2D, self).__init__()

		self.scale = scale
		self.in_channels = num_wavelength * self.scale * self.scale
		self.inter_channels = num_filter * self.scale
		
		self.unshuffle = ImageUnshuffle(self.scale)
		self.shuffle = ImageShuffle(self.scale)
		
		self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
	
		self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0)

		nn.init.constant(self.W.weight, 0)
		nn.init.constant(self.W.bias, 0)
		
		self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
		
		self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

	def forward(self, x):

		batch_size = x.size(0)
		
		x = self.unshuffle(x)
		
		g_x = self.g(x).view(batch_size, self.inter_channels, -1)
		
		g_x = g_x.permute(0,2,1)
		
		theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
		
		theta_x = theta_x.permute(0,2,1)
		
		phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
		
		f = torch.matmul(theta_x, phi_x)
	   
		f_div_C = F.softmax(f, dim=1)
		
		
		y = torch.matmul(f_div_C, g_x)
		
		y = y.permute(0,2,1).contiguous()
		 
		y = y.view(batch_size, self.inter_channels, *x.size()[2:])
		W_y = self.W(y)
		z = W_y #+ x


		return self.shuffle(z)


class MultiNonLocalBlock2D(nn.Module):
	def __init__(self):
		super(MultiNonLocalBlock2D, self).__init__()

		multihead = []
		multihead.append(NonLocalBlock2D(1))
		multihead.append(NonLocalBlock2D(2))
		multihead.append(NonLocalBlock2D(4))
		
		self.nlblk = nn.ModuleList(multihead)
	  
		self.conv = nn.Conv2d(in_channels=num_wavelength*3, out_channels=num_wavelength, kernel_size=3, stride=1, padding=1)
		
		nn.init.constant(self.conv.weight, 0)
		nn.init.constant(self.conv.bias, 0)
		

	def forward(self, x):

		res_scale1 = self.nlblk[0](x)
		res_scale2 = self.nlblk[1](x)		
		res_scale3 = self.nlblk[2](x)
		
		temp_res = torch.cat((res_scale1, res_scale2, res_scale3), 1)
		
		res = self.conv(temp_res) + x

		return res

		

class TVISTABlock(torch.nn.Module):
	def __init__(self):
		super(TVISTABlock, self).__init__()

		self.rho = nn.Parameter(torch.Tensor([0.01])) #cuda.
		self.soft_thr = nn.Parameter(torch.Tensor([0.0001])) #cuda.

	def forward(self, x):

		pad_x = nn.ReplicationPad2d((0, 0, 1, 1))(x)
		pad_y = nn.ReplicationPad2d((1, 1, 0, 0))(x)

		diff_x = pad_x[:,:,0:-2,:] - pad_x[:,:,1:-1,:]
		diff_y = pad_y[:,:,:,0:-2] - pad_y[:,:,:,1:-1]
		
		absdiff_x = torch.abs(diff_x) - self.soft_thr
		absdiff_y = torch.abs(diff_y) - self.soft_thr

		soft_x = F.relu(absdiff_x)*torch.sign(diff_x) 
		soft_y = F.relu(absdiff_x)*torch.sign(diff_x)
		
		temp_soft_x = nn.ReplicationPad2d((0, 0, 1, 1))(soft_x)
		temp_soft_y = nn.ReplicationPad2d((0, 0, 1, 1))(soft_y)
		
		diff_soft_x = temp_soft_x[:,:,2:,:] - temp_soft_x[:,:,1:-1,:]
		diff_soft_y = temp_soft_y[:,:,2:,:] - temp_soft_y[:,:,1:-1,:]

		
		x_next = -self.rho * (2*pad_x[:,:,1:-1,:] - pad_x[:,:,0:-2,:] - pad_x[:,:,2:,:]) \
		- self.rho * (2*pad_y[:,:,:,1:-1] - pad_y[:,:,:,0:-2] - pad_y[:,:,:,2:]) \
		+ x + self.rho * (diff_soft_x + diff_soft_y)

		return x_next
		

class SVDISTABlock(torch.nn.Module):
	def __init__(self):
		super(SVDISTABlock, self).__init__()

		self.in_channels = num_wavelength
		filter_size = 3	
		
		self.soft_thr = nn.Parameter(torch.Tensor([0.0001])) #cuda.

	def forward(self, x):

		batch_size = x.size(0)
		num_x = x.size(2)
		num_y = x.size(3)

		mat_x = x.view(batch_size, self.in_channels, -1)
	
		u, s, v = torch.svd(mat_x[0])
		diff_s = torch.abs(s[0:]) - self.soft_thr
		mask = torch.div(F.relu(diff_s), diff_s)
				 
		s[0:] = s[0:] * mask

		mat_x_prime = torch.matmul(torch.matmul(u, torch.diag_embed(s)), v.transpose(-2, -1))
		mat_x_prime = mat_x_prime.unsqueeze(0)
		
		for i in range(batch_size-1):
			u, s, v = torch.svd(mat_x[i+1])
			diff_s = torch.abs(s[0:]) - self.soft_thr
			mask = torch.div(F.relu(diff_s), diff_s)
						
			s[0:] = s[0:] * mask
			temp_mat_x_prime = torch.matmul(torch.matmul(u, torch.diag_embed(s)), v.transpose(-2, -1))
			mat_x_prime = torch.cat([mat_x_prime,temp_mat_x_prime.unsqueeze(0)],0)
			
		x = mat_x.view(batch_size, self.in_channels, num_x, num_y)

		return x

		
# Define  Stage
class BasicStage(torch.nn.Module):
	def __init__(self, stage_no):
		super(BasicStage, self).__init__()
		
		self.stage_num = stage_no
		onestage = []
	
		A = torch.empty(7, 1, 1, requires_grad=True, device='cuda').type(torch.FloatTensor) # .cuda
		torch.nn.init.constant_(A, 1.0)
		self.alpha = nn.Parameter(A)

		onestage.append(MultiNonLocalBlock2D()) #ISTABlock()
		onestage.append(SVDISTABlock())
		onestage.append(TVISTABlock())
		self.fcs = nn.ModuleList(onestage)

	def forward(self, I, R, Rq, Rqplus, L, Lr, Lt, Lrplus, Ltplus, M, U_Rq, U_Rqplus, U_Lr, U_Lt, U_Lrplus, U_Ltplus, U_M):#, gt_L):

		#Update R 
		inv_L = 1/(self.alpha[0] * L * L + self.alpha[1])
		R_next = inv_L*(L*(self.alpha[0]*M+U_M)+self.alpha[1]*Rq+U_Rq)
		#Update Rq 
		Rq_sim = (self.alpha[1]*R_next + self.alpha[4]*Rqplus + U_Rqplus - U_Rq)/(self.alpha[1]+self.alpha[4])
		Rq_next = self.fcs[0](Rq_sim)
		
		#Update Rqplus
		Rqplus_next = torch.min(F.relu(Rq_next - U_Rqplus/self.alpha[4]), torch.ones_like(Rq_next))

		#Update U_Rqplus		
		U_Rqplus_next = U_Rqplus + self.alpha[4]*(Rq_next - Rqplus_next)

		#Update M		
		M_next = (I + self.alpha[0]*L*R_next - U_M)/(1+self.alpha[0])
		
		#Update U_Rq
		U_Rq_next = U_Rq + self.alpha[1]*(R_next - Rq_next)
		
		#Update L
		inv_R = 1/(self.alpha[0] * R * R + self.alpha[2] + self.alpha[3])
		L_next = inv_R*(R*(self.alpha[0]*M_next+U_M)+self.alpha[2]*Lr+self.alpha[3]*Lt+U_Lr+U_Lt)
		
		#Update Lr 
		Lr_sim = (self.alpha[2]*L_next + self.alpha[5]*Lrplus + U_Lrplus - U_Lr)/(self.alpha[2]+self.alpha[5])
		Lr_next = self.fcs[1](Lr_sim)
		
		#Update Lt
		Lt_sim = (self.alpha[3]*L_next + self.alpha[6]*Ltplus + U_Ltplus - U_Lt)/(self.alpha[3]+self.alpha[6])
		Lt_next = self.fcs[2](Lt_sim)
 
		#Update Lrplus
		Lrplus_next = F.relu(Lr_next-U_Lrplus/self.alpha[5])

		#Update Ltplus
		Ltplus_next = F.relu(Lt_next-U_Ltplus/self.alpha[6])
		
		#Update U_Lrplus
		U_Lrplus_next = U_Lrplus + self.alpha[5]*(Lr_next - Lrplus_next)	

		#Update U_Ltplus
		U_Ltplus_next = U_Ltplus + self.alpha[6]*(Lt_next - Ltplus_next)
		
		#Update U_Lr
		U_Lr_next = U_Lr + self.alpha[2]*(L_next - Lr_next)		

		#Update U_Lt
		U_Lt_next = U_Lt + self.alpha[3]*(L_next - Lt_next)

		#Update U_M
		U_M_next = U_M + self.alpha[0]*(L_next*R_next - M_next)

		
		return [R_next, Rq_next, Rqplus_next, L_next, Lr_next, Lt_next, Lrplus_next, Ltplus_next, M_next, \
		U_Rq_next, U_Rqplus_next, U_Lr_next, U_Lt_next, U_Lrplus_next, U_Ltplus_next, U_M_next]
		


# Define main metwork
class CMFNET(torch.nn.Module):
	def __init__(self):
	
		super(CMFNET, self).__init__()
		layers = []
		self.LayerNo = layer_num

		for i in range(self.LayerNo):
			layers.append(BasicStage(i))

		self.fcs = nn.ModuleList(layers)

	def forward(self, gt, initial):#, gt_L):

		I = gt 

		L_p = nn.ReflectionPad2d(128)(gt) #(torch.abs(self.conv_2(self.conv_1(gt))))
		L_p = torch.nn.AvgPool2d((256,256))(L_p)
			
		L = nn.Upsample(scale_factor=128, mode='bilinear', align_corners=True)(L_p)

		I = (ImageUnshuffle(4)(I)).view(-1, num_wavelength, int(num_x/4), int(num_y/4))		
		L = (ImageUnshuffle(4)(L)).view(-1, num_wavelength, int(num_x/4), int(num_y/4))
		Lr = L
		Lt = L
		Lrplus = L
		Ltplus = L
		
		R = torch.min((I + 1e-10) / (L + 1e-10),torch.ones_like(L))
		Rq = R
		Rqplus = R
		
		M = L*R

		U_Rq = R - R
		U_Rqplus = R - R

		U_Lr = L - L
		U_Lt = L - L
		U_Lrplus = L - L
		U_Ltplus = L - L
		
		U_M = I - M
		
		for i in range(self.LayerNo):
			[R, Rq, Rqplus, L, Lr, Lt, Lrplus, Ltplus, M, U_Rq, U_Rqplus, U_Lr, U_Lt, U_Lrplus, U_Ltplus, U_M] \
			= self.fcs[i](I, R, Rq, Rqplus, L, Lr, Lt, Lrplus, Ltplus, M, U_Rq, U_Rqplus,  U_Lr, U_Lt, U_Lrplus, U_Ltplus, U_M)#, gt_L)

		res_L = (ImageShuffle(4)(Lrplus.view(-1, 4*4*num_wavelength, int(num_x/4), int(num_y/4)))).view(-1, num_wavelength, num_x, num_y)
		res_R = (ImageShuffle(4)(Rqplus.view(-1, 4*4*num_wavelength, int(num_x/4), int(num_y/4)))).view(-1, num_wavelength, num_x, num_y)

		return res_R, res_L


def count_parameters(model):
	print('parameters amount:')
	return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_Initial(batch_data, num_initial, method):
	
	batch_size = batch_data.shape[0]
	batch_initial = np.zeros((batch_size, 31*num_initial, size_img, size_img))
	maxsize = [int(128/128), int(96/128), int(64/128)]
	gaussiansigma = [3, 2, 1]
	for i in range(batch_size):
		for j in range(31):
			for k in range(num_initial):
				temp = batch_data[i, j, : :]
				#print(temp.shape)
				if method == 'max':
					batch_initial[i, k*31+j, :, :] = 0.99*maximum_filter(temp, maxsize[k]) + 0.01*median_filter(temp, maxsize[k])
					#batch_initial[i, k*31+j, :, :] = batch_initial[i, k*31+j, :, :]/np.max(batch_initial[i, k*31+j, :, :])
				elif method == 'gaussian':
					 temp = gaussian_filter(temp, gaussiansigma[k])
					 #print(temp.shape)
					 temp = np.power(np.sum(np.abs(np.gradient(temp)),axis = 0),3)
					 #print(temp.shape)
					 temp = np.power( uniform_filter(temp,45),1/3)
					 #print(temp.shape)
					 batch_initial[i, k*31+j, :, :] = temp/np.max(temp) #np.power(gaussian_filter(temp, 5), 1)
	print(np.max(batch_initial))
	print(np.max(batch_data))
	return batch_initial

	
print("Program starts.")
	
model = CMFNET()

#model = model.cuda()

#print(model)

model.load_state_dict(torch.load('./model/net_params.pkl'))

print(count_parameters(model))

model.eval()


img_name = args.imgname
mat = h5py.File(img_name + '.mat')
test_x = mat['image']
test_x.astype(np.float32)
print(test_x.shape)
#test_x = test_x[0:512:2, 0:512:2, 0:31, :]

test_x = np.transpose(test_x, (3, 2, 1, 0))

test_x = test_x / np.max(test_x) 

print(test_x.shape)

batch_initial = get_Initial(test_x, 1, 'max')


print(batch_initial.shape)
test_x = torch.from_numpy(test_x).float()#.to(device)
batch_initial = torch.from_numpy(batch_initial).float()#.to(device)

ref_output, light_output = model(test_x, batch_initial)

prec_light = light_output.cpu().detach().numpy()
#prec_ref = ref_output.cpu().detach().numpy()
test_data = test_x.cpu().detach().numpy()

sio.savemat('res_'+img_name+'.mat', {'predict_light':prec_light})
