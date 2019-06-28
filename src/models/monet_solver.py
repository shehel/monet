import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image
from tqdm import tqdm 
import pdb
import visdom
from models.utils import cuda, grid2gif
from data.clevr_dataset import return_data
from models.monet_model import MONet

import os 


def reconstruction_loss(x, x_recons, mask, distribution):
    #pdb.set_trace()
    batch_size = x.size(0)
    assert batch_size != 0
    #_trdb.set_trace()
    
    #pdb.set_trace()
    #mask = F.normalize(mask)
    #mask_recon = F.tanh(mask_recon)
    #mask_recon_loss = F.binary_cross_entropy_with_logits(mask, mask_recon, size_average=False).div(batch_size)
    #mask_recon_loss = F.mse_loss(mask, mask_recon, size_average=False).div(batch_size)    
    recon_loss = 0
    for idx, i in enumerate(x_recons):
        imgs = F.sigmoid(i)
        #pdb.set_trace()
        m = mask[:,idx,:,:]
        m = torch.reshape(m, (m.shape[0], 1, m.shape[1], m.shape[2]))
        recon_loss = F.mse_loss(torch.mul(imgs, m), torch.mul(x, m), size_average=False).div(batch_size) + recon_loss


    return recon_loss


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld

class DataGather(object):
    def __init__(self):
        self.data = self.get_empty_data_dict()

    def get_empty_data_dict(self):
        return dict(iter=[],
                    recon_loss=[],
                    total_kld=[],
                    dim_wise_kld=[],
                    mean_kld=[],
                    mu=[],
                    var=[],
                    images=[],)

    def insert(self, **kwargs):
        for key in kwargs:
            self.data[key].append(kwargs[key])

    def flush(self):
        self.data = self.get_empty_data_dict()

class MONet_Solver(object):
    def __init__(self, args):
        self.use_cuda = args.cuda and torch.cuda.is_available()
        self.max_iter = args.max_iter
        self.global_iter = 0


        self.steps = 5
        self.z_dim = args.z_dim
        self.beta = args.beta
        self.gamma = args.gamma
        self.C_max = args.C_max
        self.C_stop_iter = args.C_stop_iter
        self.objective = args.objective
        self.model = args.model
        self.lr = args.lr
        self.beta1 = args.beta1
        self.beta2 = args.beta2

        #if args.dataset.lower() == 'room':
        self.nc = 3
        self.decoder_dist = 'gaussian'
        #else:
        #    raise NotImplementedError

        self.viz_name = args.viz_name
        self.viz_port = args.viz_port
        self.viz_on = args.viz_on
        self.win_recon = None
        self.win_kld = None
        self.win_mu = None
        self.win_var = None
        if self.viz_on:
            self.viz = visdom.Visdom(port=self.viz_port)
        
        self.net = cuda(MONet(), self.use_cuda)
        self.optim = optim.RMSprop(self.net.parameters(), lr=self.lr)
        self.ckpt_dir = os.path.join(args.ckpt_dir, args.viz_name)
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir, exist_ok=True)
        self.ckpt_name = args.ckpt_name
        if self.ckpt_name is not None:
            self.load_checkpoint(self.ckpt_name)

        self.save_output = args.save_output
        self.output_dir = os.path.join(args.output_dir, args.viz_name)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

       
        self.dset_dir = args.dset_dir
        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.data_loader = return_data(self.dset_dir, self.batch_size)
        #self.data_loader = return_data(args)
        self.gather_step = args.gather_step
        self.display_step = args.display_step
        self.save_step = args.save_step

        self.gather = DataGather()
   

    def train(self):

        self.net_mode(train=True)
        self.C_max = Variable(cuda(torch.FloatTensor([self.C_max]), self.use_cuda))

        pbar = tqdm(total=self.max_iter)
        pbar.update(self.global_iter)
        out = False


        while not out:
            for x in self.data_loader:
                self.global_iter += 1
                pbar.update(1)
                
                x = Variable(cuda(x, self.use_cuda))
                s_i = (cuda(torch.zeros(self.batch_size, 1, 128, 128),self.use_cuda))
                loss = 0
                recons = []
                kld = 0
                for k in range(self.steps):
                    #x = torch.cat((x, s_i), 1)
                    #if self.global_iter == 31:
                    #pdb.set_trace() 
                    if k == self.steps - 1:
                        s_i, mask, recon, mu, logvar = self.net(x, s_i, True)
                    else:
                        s_i, mask, recon, mu, logvar = self.net(x, s_i)
                    x_recon, mask_recon = torch.split(recon, (3,1), 1)
                             
                    #recon_loss, mask_recon_loss = reconstruction_loss(x, mask, x_recon,
                    #                                mask_recon, self.decoder_dist)   
                    #pdb.set_trace()
                    #mask_recon = F.logsigmoid(mask_recon)
                    #mask_recon_loss = F.kl_div(mask_recon, mask, size_average=False).div(4)
                    
                    
                    total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
                    kld = total_kld+kld
                    #loss += recon_loss + self.beta*total_kld + 0.5*mask_recon_loss
                    recons.append(x_recon)
                    if k>0:
                        masks = torch.cat((masks, mask), dim=1)
                        recon_masks = torch.cat((recon_masks, mask_recon), dim = 1)
                    else:
                        masks = mask
                        recon_masks = mask_recon
                soft_mask = F.softmax(masks, dim = 1)
                soft_recon = F.softmax(recon_masks, dim = 1)
                mask_loss = F.mse_loss(soft_mask, soft_recon, size_average=False).div(soft_mask.size(0))
                recon_loss = reconstruction_loss(x, recons, masks, "gaussian")
                if self.viz_on and self.global_iter%self.gather_step == 0:
                    self.gather.insert(iter=self.global_iter,
                                    mu=mu.mean(0).data, var=logvar.exp().mean(0).data,
                                    recon_loss=recon_loss.data, total_kld=total_kld.data,
                                    dim_wise_kld=dim_wise_kld.data, mean_kld=mean_kld.data)
                        
                if self.global_iter%self.display_step == 0:
                    pbar.write('[{}] img recon_loss: {:.3f} mask_recon_loss:{:.3f} total_kld:{:.3f} mean_kld:{:.3f}'.format(
                        self.global_iter, recon_loss.item(), mask_loss.item(), total_kld.item(), mean_kld.item()))

                    var = logvar.exp().mean(0).data
                    var_str = ''
                    for j, var_j in enumerate(var):
                        var_str += 'var{}:{:.4f} '.format(j+1, var_j)
                    pbar.write(var_str)

                    if self.objective == 'B' or self.objective == 'S':
                        pbar.write('C:{:.3f}'.format(C.data[0]))
                    
                    if self.viz_on:
                        #print (self.global_iter)
                        #pdb.set_trace()
                        j = masks[:, 0, :, :]
                        j = j.reshape((j.shape[0], 1, j.shape[1], j.shape[2]))
                        self.gather.insert(images=x)
                        self.gather.insert(images=soft_mask[:, 0, :, :])
                        self.gather.insert(images=soft_mask[:, 1, :, :])
                        #self.gather.insert(images=F.sigmoid(recons[0]))
                        self.gather.insert(images=soft_mask[:, 2, :, :])
                        self.gather.insert(images=soft_mask[:, 3, :, :])

                        #self.gather.insert(images=torch.mul(F.sigmoid(recons[0]), j))
                        self.viz_reconstruction(k)
                        self.viz_lines()
                        self.gather.flush()

                   #if self.viz_on or self.save_output:
                   #     self.viz_traverse()
                #pdb.set_trace()
                self.optim.zero_grad()
                #pdb.set_trace()
                loss += recon_loss + self.beta*kld + 0.5*mask_loss
                loss.backward()
                self.optim.step() 

                if self.global_iter%self.save_step == 0:
                   self.save_checkpoint('last')
                   pbar.write('Saved checkpoint(iter:{})'.format(self.global_iter))

                if self.global_iter%50000 == 0:
                   self.save_checkpoint(str(self.global_iter))

                if self.global_iter >= self.max_iter:
                    out = True
                    break
        
        pbar.write("[Training Finished]")
        pbar.close()


    def viz_reconstruction(self, k=0):
        self.net_mode(train=False)
        #pdb.set_trace()
        x = self.gather.data['images'][0][:1]
        x = make_grid(x, normalize=False)
        scope = self.gather.data['images'][1][:1]
        scope = make_grid(scope, normalize=False)
        mask = self.gather.data['images'][2][:1]
        mask = make_grid(mask, normalize=False)
        x_recon = self.gather.data['images'][3][:1]
        x_recon = make_grid(x_recon, normalize=False)
        mask_recon = self.gather.data['images'][4][:1]
        mask_recon = make_grid(mask_recon, normalize=False)
        images = torch.stack([x, scope, mask, x_recon, mask_recon], dim=0).cpu()
        self.viz.images(images, env=self.viz_name+'_reconstruction',
                        opts=dict(title=str(self.global_iter)+str(k)), nrow=1)
        self.net_mode(train=True)

    def viz_lines(self):
        self.net_mode(train=False)
        recon_losses = torch.stack(self.gather.data['recon_loss']).cpu()

        mus = torch.stack(self.gather.data['mu']).cpu()
        vars = torch.stack(self.gather.data['var']).cpu()

        dim_wise_klds = torch.stack(self.gather.data['dim_wise_kld'])
        mean_klds = torch.stack(self.gather.data['mean_kld'])
        total_klds = torch.stack(self.gather.data['total_kld'])
        klds = torch.cat([dim_wise_klds, mean_klds, total_klds], 1).cpu()
        iters = torch.Tensor(self.gather.data['iter'])

        legend = []
        for z_j in range(self.z_dim):
            legend.append('z_{}'.format(z_j))
        legend.append('mean')
        legend.append('total')

        if self.win_recon is None:
            self.win_recon = self.viz.line(
                                        X=iters,
                                        Y=recon_losses,
                                        env=self.viz_name+'_lines',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            xlabel='iteration',
                                            title='reconsturction loss',))
        else:
            self.win_recon = self.viz.line(
                                        X=iters,
                                        Y=recon_losses,
                                        env=self.viz_name+'_lines',
                                        win=self.win_recon,
                                        update='append',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            xlabel='iteration',
                                            title='reconsturction loss',))

        if self.win_kld is None:
            self.win_kld = self.viz.line(
                                        X=iters,
                                        Y=klds,
                                        env=self.viz_name+'_lines',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            legend=legend,
                                            xlabel='iteration',
                                            title='kl divergence',))
        else:
            self.win_kld = self.viz.line(
                                        X=iters,
                                        Y=klds,
                                        env=self.viz_name+'_lines',
                                        win=self.win_kld,
                                        update='append',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            legend=legend,
                                            xlabel='iteration',
                                            title='kl divergence',))

        if self.win_mu is None:
            self.win_mu = self.viz.line(
                                        X=iters,
                                        Y=mus,
                                        env=self.viz_name+'_lines',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            legend=legend[:self.z_dim],
                                            xlabel='iteration',
                                            title='posterior mean',))
        else:
            self.win_mu = self.viz.line(
                                        X=iters,
                                        Y=vars,
                                        env=self.viz_name+'_lines',
                                        win=self.win_mu,
                                        update='append',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            legend=legend[:self.z_dim],
                                            xlabel='iteration',
                                            title='posterior mean',))

        if self.win_var is None:
            self.win_var = self.viz.line(
                                        X=iters,
                                        Y=vars,
                                        env=self.viz_name+'_lines',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            legend=legend[:self.z_dim],
                                            xlabel='iteration',
                                            title='posterior variance',))
        else:
            self.win_var = self.viz.line(
                                        X=iters,
                                        Y=vars,
                                        env=self.viz_name+'_lines',
                                        win=self.win_var,
                                        update='append',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            legend=legend[:self.z_dim],
                                            xlabel='iteration',
                                            title='posterior variance',))
        self.net_mode(train=True)
    
    def net_mode(self, train):
        if not isinstance(train, bool):
            raise('Only bool type is supported. True or False')

        if train:
            self.net.train()
        else:
            self.net.eval()

    def save_checkpoint(self, filename, silent=True):
        model_states = {'net':self.net.state_dict(),}
        optim_states = {'optim':self.optim.state_dict(),}
        win_states = {'recon':self.win_recon,
                      'kld':self.win_kld,
                      'mu':self.win_mu,
                      'var':self.win_var,}
        states = {'iter':self.global_iter,
                  'win_states':win_states,
                  'model_states':model_states,
                  'optim_states':optim_states}

        file_path = os.path.join(self.ckpt_dir, filename)
        with open(file_path, mode='wb+') as f:
            torch.save(states, f)
        if not silent:
            print("=> saved checkpoint '{}' (iter {})".format(file_path, self.global_iter))

    def load_checkpoint(self, filename):
        file_path = os.path.join(self.ckpt_dir, filename)
        if os.path.isfile(file_path):
            checkpoint = torch.load(file_path)
            self.global_iter = checkpoint['iter']
            self.win_recon = checkpoint['win_states']['recon']
            self.win_kld = checkpoint['win_states']['kld']
            self.win_var = checkpoint['win_states']['var']
            self.win_mu = checkpoint['win_states']['mu']
            self.net.load_state_dict(checkpoint['model_states']['net'])
            self.optim.load_state_dict(checkpoint['optim_states']['optim'])
            print("=> loaded checkpoint '{} (iter {})'".format(file_path, self.global_iter))
        else:
            print("=> no checkpoint found at '{}'".format(file_path))