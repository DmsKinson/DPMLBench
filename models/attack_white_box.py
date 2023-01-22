import torch
import torch.nn as nn

class WhiteBoxAttackModel(nn.Module):
	def __init__(self, class_num, total):
		super(WhiteBoxAttackModel, self).__init__()

		self.Output_Component = nn.Sequential(
			nn.Dropout(p=0.2),
			nn.Linear(class_num, 128),
			nn.ReLU(),
			nn.Linear(128, 64),
		)

		self.Loss_Component = nn.Sequential(
			nn.Dropout(p=0.2),
			nn.Linear(1, 128),
			nn.ReLU(),
			nn.Linear(128, 64),
		)

		self.Gradient_Component = nn.Sequential(
			nn.Dropout(p=0.2),
			nn.Conv2d(1, 1, kernel_size=5, padding=2),
			nn.BatchNorm2d(1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2),
			nn.Flatten(),
			nn.Dropout(p=0.2),
			nn.Linear(total, 256),
			nn.ReLU(),
			nn.Dropout(p=0.2),
			nn.Linear(256, 128),
			nn.ReLU(),
			nn.Linear(128, 64),
		)

		self.Label_Component = nn.Sequential(
			nn.Dropout(p=0.2),
			nn.Linear(class_num, 128),
			nn.ReLU(),
			nn.Linear(128, 64),
		)

		self.Encoder_Component = nn.Sequential(
			nn.Dropout(p=0.2),
			nn.Linear(256, 256),
			nn.ReLU(),
			nn.Dropout(p=0.2),
			nn.Linear(256, 128),
			nn.ReLU(),
			nn.Dropout(p=0.2),
			nn.Linear(128, 64),
			nn.ReLU(),
			nn.Linear(64, 2),
		)


	def forward(self, output, loss, gradient, label):
		Output_Component_result = self.Output_Component(output)
		Loss_Component_result = self.Loss_Component(loss)
		Gradient_Component_result = self.Gradient_Component(gradient)
		Label_Component_result = self.Label_Component(label)
		
		final_inputs = torch.cat((Output_Component_result, Loss_Component_result, Gradient_Component_result, Label_Component_result), 1)
		final_result = self.Encoder_Component(final_inputs)

		return final_result


class WhiteBoxAttackModelBinary(nn.Module):
	def __init__(self, class_num, kernel_size, layer_size):
		'''
		class_num: num of dimension of target output
		kernel_size_list: list of integers containing the input size of FC layers in the
        target model, whose gradient will be fed into the "gradient component"
		'''
		super(WhiteBoxAttackModelBinary, self).__init__()
		self.num_filters = 10
		self.Output_Component = nn.Sequential(
			nn.Linear(class_num, 128),
			nn.ReLU(),
			nn.Dropout(p=0.2),
			nn.Linear(128, 64),
			nn.ReLU(),
			nn.Dropout(p=0.2)
		)

		self.Loss_Component = nn.Sequential(
			nn.Linear(1, 128),
			nn.ReLU(),
			nn.Dropout(p=0.2),
			nn.Linear(128, 64),
			nn.ReLU(),
			nn.Dropout(p=0.2),
		)
		self.Gradient_Components = []

		self.Gradient_Component = nn.Sequential(
			nn.Conv2d(1, self.num_filters, kernel_size=[1,layer_size], stride=1),
			nn.ReLU(),
			nn.Dropout(p=0.2),
			nn.Flatten(),
			nn.Linear(self.num_filters*kernel_size, 128),
			nn.ReLU(),
			nn.Dropout(p=0.2),
			nn.Linear(128, 64),
			nn.ReLU(),
			nn.Dropout(p=0.2)
		)
			

		self.Label_Component = nn.Sequential(
			nn.Linear(class_num, 128),
			nn.ReLU(),
			nn.Dropout(p=0.2),
			nn.Linear(128, 64),
			nn.ReLU(),
			nn.Dropout(p=0.2)
		)

		# output of label,output,loss is 64
		self.Encoder_Component = nn.Sequential(
			nn.Linear(256, 256),
			nn.ReLU(),
			nn.Dropout(p=0.2),
			nn.Linear(256, 128),
			nn.ReLU(),
			nn.Dropout(p=0.2),
			nn.Linear(128, 64),
			nn.ReLU(),
			nn.Dropout(p=0.2),
			nn.Linear(64, 1),
		)


	def forward(self, output, loss, gradient, label):
		Output_Component_result = self.Output_Component(output)
		Loss_Component_result = self.Loss_Component(loss)

		Grad_Component_result = self.Gradient_Component(gradient)
		Label_Component_result = self.Label_Component(label)
		
		final_inputs = torch.cat((Output_Component_result, Loss_Component_result, Grad_Component_result, Label_Component_result), 1)
		final_result = self.Encoder_Component(final_inputs)

		return final_result