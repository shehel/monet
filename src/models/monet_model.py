import torch
from models.model import BetaVAE_H, BetaVAE_B, Spatial_Decoder, BetaVAE_paper
from models.unet_model import UNet
from torch import nn
from torch.nn import functional as F
import pdb
class MONet(nn.Module):
	def __init__(self, z_dim = 16, nc = 3):
		super(MONet,self).__init__()
		self.z_dim = z_dim
		self.nc = nc
		self.component_vae = Spatial_Decoder(self.z_dim, self.nc+1)
		self.attention_net = UNet(self.nc+1, 1)

	def forward(self, x, scope, last = False):
		if last:
			mask = scope
		else:
			inp_attention = torch.cat((x, scope), 1)
			logits  = self.attention_net(inp_attention)
			#pdb.set_trace()
			mask = scope + F.log_softmax(logits)
			scope = scope + (1-F.log_softmax(logits))
		inp_vae = torch.cat((x, mask), 1)
		x_recon, mu, log_var = self.component_vae(inp_vae)
		#pdb.set_trace()

		return scope, mask, x_recon, mu, log_var