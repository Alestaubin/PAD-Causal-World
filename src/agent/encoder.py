import torch
import torch.nn as nn
import torch.nn.functional as F

OUT_DIM = {2: 39, 4: 35, 6: 31, 8: 27, 10: 23, 11: 21, 12: 19}


def tie_weights(src, trg):
	assert type(src) == type(trg)
	trg.weight = src.weight
	trg.bias = src.bias


class CenterCrop(nn.Module):
	"""Center-crop if observation is not already cropped"""
	def __init__(self, size):
		super().__init__()
		assert size == 84
		self.size = size

	def forward(self, x):
		assert x.ndim == 4, 'input must be a 4D tensor'
		if x.size(2) == self.size and x.size(3) == self.size:
			return x
		elif x.size(-1) == 100:
			return x[:, :, 8:-8, 8:-8]
		else:
			return ValueError('unexpected input size')


class NormalizeImg(nn.Module):
	"""Normalize observation"""
	def forward(self, x):
		return x/255.


class PixelEncoder(nn.Module):
	"""Convolutional encoder of pixel observations"""
	def __init__(self, obs_shape, feature_dim, num_layers=4, num_filters=32, num_shared_layers=4):
		super().__init__()
		assert len(obs_shape) == 3

		self.feature_dim = feature_dim
		self.num_layers = num_layers
		self.num_shared_layers = num_shared_layers

		self.preprocess = nn.Sequential(
			CenterCrop(size=84), NormalizeImg()
		)

		self.convs = nn.ModuleList(
			[nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
		)
		for i in range(num_layers - 1):
			self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

		out_dim = OUT_DIM[num_layers]
		self.fc = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)
		self.ln = nn.LayerNorm(self.feature_dim)

	def forward_conv(self, obs, detach=False):
		obs = self.preprocess(obs)
		conv = torch.relu(self.convs[0](obs))

		for i in range(1, self.num_layers):
			conv = torch.relu(self.convs[i](conv))
			if i == self.num_shared_layers-1 and detach:
				conv = conv.detach()

		h = conv.view(conv.size(0), -1)
		return h

	def forward(self, obs, detach=False):
		h = self.forward_conv(obs, detach)
		h_fc = self.fc(h)
		h_norm = self.ln(h_fc)
		out = torch.tanh(h_norm)

		return out

	def copy_conv_weights_from(self, source, n=None):
		"""Tie n first convolutional layers"""
		if n is None:
			n = self.num_layers
		for i in range(n):
			tie_weights(src=source.convs[i], trg=self.convs[i])
			
class IdentityEncoder(nn.Module):
	"""
	Deep encoder for 1D vector observations.
	Projects the state vector to the hidden feature dimension using an MLP.
	"""
	def __init__(self, obs_shape, feature_dim, num_layers, num_shared_layers, *args):
		super().__init__()
		assert len(obs_shape) == 1
		self.feature_dim = feature_dim
		self.num_layers = num_layers
		
		# Define a stack of Linear layers (MLP)
		self.layers = nn.ModuleList()
		
		# First layer: Input dimension -> Feature Dimension
		self.layers.append(nn.Linear(obs_shape[0], feature_dim))
		
		# Subsequent layers: Feature Dimension -> Feature Dimension
		# This makes the encoder "deeper" based on num_layers
		for _ in range(num_layers - 1):
			self.layers.append(nn.Linear(feature_dim, feature_dim))

		self.ln = nn.LayerNorm(feature_dim)
		self.num_shared_layers = num_shared_layers  

	def forward(self, obs, detach=False):
		# obs shape: (batch_size, obs_dim)
		out = obs
		
		# Pass through the MLP layers
		for i, layer in enumerate(self.layers):
			out = layer(out)
			# Apply ReLU after every layer except the last one 
			if i < len(self.layers) - 1:
				out = F.relu(out)

			if detach and (i == self.num_shared_layers - 1):
				out = out.detach()
		
		# Final processing
		out = self.ln(out)
		out = torch.tanh(out) # Tanh to match PixelEncoder distribution

		if detach:
			out = out.detach()
		return out

	def copy_conv_weights_from(self, source, n=None):
		"""
		Tie the weights of the MLP layers.
		Even though these aren't convolutions, we maintain the naming 
		convention to stay compatible with the Agent's API.
		"""
		if n is None:
			n = self.num_layers
		
		# Tie the weights for the first n layers
		for i in range(n):
			print(" Tying layer ", i)
			if i < len(self.layers) and i < len(source.layers):
				tie_weights(src=source.layers[i], trg=self.layers[i])

class IdentityEncoder2(nn.Module):
	"""
	Deep encoder for 1D vector observations.
	Projects the state vector to the hidden feature dimension using an MLP.
	"""
	def __init__(self, obs_shape, feature_dim, num_layers, num_shared_layers, *args):
		super().__init__()
		assert len(obs_shape) == 1
		self.feature_dim = feature_dim
		self.num_layers = num_layers
		
		# Define a stack of Linear layers (MLP)
		self.layers = nn.ModuleList()
		self.norms = nn.ModuleList()
		
		# First layer: Input dimension -> Feature Dimension
		self.layers.append(nn.Linear(obs_shape[0], feature_dim))
		self.norms.append(nn.LayerNorm(feature_dim))
		
		for _ in range(num_layers - 1):
			self.layers.append(nn.Linear(feature_dim, feature_dim))
			self.norms.append(nn.LayerNorm(feature_dim))

		self.num_shared_layers = num_shared_layers  

	def forward(self, obs, detach=False):
		# obs shape: (batch_size, obs_dim)
		out = obs
		
		# Pass through the MLP layers
		for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):	
			out = layer(out)
			out = norm(out)

			# Apply ReLU after every layer except the last one 
			if i < len(self.layers) - 1:
				out = F.relu(out)

			if detach and (i == self.num_shared_layers - 1):
				out = out.detach()
		
		out = torch.tanh(out) # Tanh to match PixelEncoder distribution

		if detach:
			out = out.detach()
		return out

	def copy_conv_weights_from(self, source, n=None):
		"""
		Tie the weights of the MLP layers.
		Even though these aren't convolutions, we maintain the naming 
		convention to stay compatible with the Agent's API.
		"""
		if n is None:
			n = self.num_layers
		
		# Tie the weights for the first n layers
		for i in range(n):
			print(" Tying layer ", i)
			if i < len(self.layers) and i < len(source.layers):
				tie_weights(src=source.layers[i], trg=self.layers[i])
				tie_weights(src=source.norms[i], trg=self.norms[i])


def make_encoder(
    obs_type, obs_shape, feature_dim, num_layers, num_filters, num_shared_layers
):
    if obs_type == 'pixel':
        assert num_layers in OUT_DIM.keys(), 'invalid number of layers'
        if num_shared_layers == -1 or num_shared_layers == None:
            num_shared_layers = num_layers
        assert num_shared_layers <= num_layers and num_shared_layers > 0, \
            f'invalid number of shared layers, received {num_shared_layers} layers'
        
        return PixelEncoder(
            obs_shape, feature_dim, num_layers, num_filters, num_shared_layers
        )
        
    elif obs_type == 'structured':
        return IdentityEncoder2(
            obs_shape, feature_dim, num_layers, num_filters, num_shared_layers
        )
    
    else:
        raise ValueError(f"Invalid obs_type: {obs_type}")