import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from models.layers.mesh_conv import MeshConv
import torch.nn.functional as F
from models.layers.mesh_pool import MeshPool
from models.layers.mesh_unpool import MeshUnpool
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

import sys 
sys.path.append('/home/students/jlee/libs/chamferdist')
from chamferdist import ChamferDistance

###############################################################################
# Helper Functions
###############################################################################


def get_norm_layer(norm_type='instance', num_groups=1):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'group':
        norm_layer = functools.partial(nn.GroupNorm, affine=True, num_groups=num_groups)
    elif norm_type == 'none':
        norm_layer = NoNorm
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def get_norm_args(norm_layer, nfeats_list):
    if hasattr(norm_layer, '__name__') and norm_layer.__name__ == 'NoNorm':
        norm_args = [{'fake': True} for f in nfeats_list]
    elif norm_layer.func.__name__ == 'GroupNorm':
        norm_args = [{'num_channels': f} for f in nfeats_list]
    elif norm_layer.func.__name__ == 'BatchNorm':
        norm_args = [{'num_features': f} for f in nfeats_list]
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_layer.func.__name__)
    return norm_args

class NoNorm(nn.Module): #todo with abstractclass and pass
    def __init__(self, fake=True):
        self.fake = fake
        super(NoNorm, self).__init__()
    def forward(self, x):
        return x
    def __call__(self, x):
        return self.forward(x)

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type, init_gain):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    net.apply(init_func)


def init_net(net, init_type, init_gain, gpu_ids):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.cuda(gpu_ids[0])
        net = net.cuda()
        net = torch.nn.DataParallel(net, gpu_ids)
    if init_type != 'none':
        init_weights(net, init_type, init_gain)
    return net


def define_classifier(input_nc, ncf, ninput_edges, nclasses, opt, gpu_ids, arch, init_type, init_gain):
    net = None
    norm_layer = get_norm_layer(norm_type=opt.norm, num_groups=opt.num_groups)

    if arch == 'mconvnet':
        net = MeshConvNet(norm_layer, input_nc, ncf, nclasses, ninput_edges, opt.pool_res, opt.fc_n,
                          opt.resblocks)
    elif arch == 'meshunet':
        down_convs = [input_nc] + ncf
        up_convs = ncf[::-1] + [nclasses]
        pool_res = [ninput_edges] + opt.pool_res
        net = MeshEncoderDecoder(pool_res, down_convs, up_convs, blocks=opt.resblocks,
                                 transfer_data=True)
        print('meshunet is created')
    elif arch == 'meshae':
        net = MeshAutoEncoder(norm_layer, input_nc, ncf, nclasses, ninput_edges, opt.pool_res, opt.fc_n,
                          opt.resblocks)
    else:
        raise NotImplementedError('Encoder model name [%s] is not recognized' % arch)
    return init_net(net, init_type, init_gain, gpu_ids)

def define_loss(opt):
    if opt.dataset_mode == 'classification':
        # loss = torch.nn.NLLLoss()
        loss = torch.nn.CrossEntropyLoss()
    elif opt.dataset_mode == 'segmentation':
        loss = torch.nn.CrossEntropyLoss(ignore_index=-1)
    elif opt.dataset_mode == 'autoencoder':
        #loss = torch.nn.L1Loss()
        loss = ChamferDistance()
    return loss

##############################################################################
# Classes For Classification / Segmentation Networks
##############################################################################

class MeshConvNet(nn.Module):
    """Network for learning a global shape descriptor (classification)
    """
    def __init__(self, norm_layer, nf0, conv_res, nclasses, input_res, pool_res, fc_n,
                 nresblocks=3):
        super(MeshConvNet, self).__init__()
        self.k = [nf0] + conv_res
        self.res = [input_res] + pool_res
        norm_args = get_norm_args(norm_layer, self.k[1:])

        for i, ki in enumerate(self.k[:-1]):
            setattr(self, 'conv{}'.format(i), MResConv(ki, self.k[i + 1], nresblocks))
            setattr(self, 'norm{}'.format(i), norm_layer(**norm_args[i]))
            setattr(self, 'pool{}'.format(i), MeshPool(self.res[i + 1]))


        self.gp = torch.nn.AvgPool1d(self.res[-1])
        # self.gp = torch.nn.MaxPool1d(self.res[-1])
        self.fcs = []
        self.fcs.append(nn.Linear(self.k[-1], fc_n[0]))
        for i in range(len(fc_n)-1) :
            self.fcs.append(nn.Linear(fc_n[i], fc_n[i+1]))
        self.fcs = nn.ModuleList(self.fcs)
        self.last_fc = nn.Linear(fc_n[-1], nclasses)

    def forward(self, x, mesh, writer=False, steps=None):
        #print(len(mesh[0].vs))
        x_max, x_min = max(mesh[0].vs[:,0]), min(mesh[0].vs[:,0])
        y_max, y_min = max(mesh[0].vs[:,1]), min(mesh[0].vs[:,1])
        z_max, z_min = max(mesh[0].vs[:,2]), min(mesh[0].vs[:,2])
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(mesh[0].vs[:,0], mesh[0].vs[:,1], mesh[0].vs[:,2], marker='.')
        ax.set_xlim((x_min, x_max)); ax.set_ylim((y_min, y_max)); ax.set_zlim((z_min, z_max));
        # plt.savefig('/home/students/jlee/repos/meshcnn/original_mesh.png', bbox_inches='tight')
        if writer is not False:
            writer.add_figure('orignal/mesh', fig, steps)
            vsfs = mesh[0].get_vs_fs()
            writer.add_mesh('original/pointcloud', vertices=vsfs['vs'], global_step=steps)
            writer.add_mesh('origianl/mesh', vertices=vsfs['vs'], faces=vsfs['fs'], global_step=steps)
        plt.close()
        
        for i in range(len(self.k) - 1):
            x = getattr(self, 'conv{}'.format(i))(x, mesh)
            x = F.relu(getattr(self, 'norm{}'.format(i))(x))
            x = getattr(self, 'pool{}'.format(i))(x, mesh)
            vs = mesh[0].vs[mesh[0].v_mask]
            #print(len(vs))
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            ax.scatter3D(vs[:,0], vs[:,1], vs[:,2], marker='.')
            ax.set_xlim((x_min, x_max)); ax.set_ylim((y_min, y_max)); ax.set_zlim((z_min, z_max));
            # plt.savefig('/home/students/jlee/repos/meshcnn/pooled_mesh_%d.png'%(i), bbox_inches='tight')
            if writer is not False :
                writer.add_figure('pooled_mesh/%d'%(i+1), fig, steps)
                vs, fs = mesh[0].get_vs_fs()
                writer.add_mesh('pooled_pointcloud/%d'%(i+1), vertices=vsfs['vs'], global_step=steps)
                writer.add_mesh('pooled_mesh/%d'%(i+1), vertices=vsfs['vs'], faces=vsfs['fs'], global_step=steps)
            plt.close()    

        x = self.gp(x)
        x = x.view(-1, self.k[-1])

        for fc in self.fcs :
            x = F.relu(fc(x))
        x = self.last_fc(x)
       # print (x.shape, x)
        #x = F.softmax(x)
        #print (x.shape, x)
        return x
    
class MeshAutoEncoder(nn.Module):
    """Network for learning a global shape descriptor (classification)
    """
    def __init__(self, norm_layer, nf0, conv_res, nclasses, input_res, pool_res, fc_n,
                 nresblocks=3):
        super(MeshAutoEncoder, self).__init__()
        self.k = [nf0] + conv_res
        self.res = [input_res] + pool_res
        self.fc_n = fc_n
        norm_args = get_norm_args(norm_layer, self.k[1:])

        for i, ki in enumerate(self.k[:-1]):
            setattr(self, 'conv{}'.format(i), MResConv(ki, self.k[i + 1], nresblocks))
            setattr(self, 'norm{}'.format(i), norm_layer(**norm_args[i]))
            setattr(self, 'pool{}'.format(i), MeshPool(self.res[i + 1]))


        # self.gp = torch.nn.AvgPool1d(self.res[-1])
        self.gp = torch.nn.MaxPool1d(self.res[-1])
        self.decoder_fc1 = nn.Linear(self.k[-1], fc_n[0])
        self.decoder_fc2 = nn.Linear(fc_n[0], fc_n[1])
        self.decoder_fc3 = nn.Linear(fc_n[1], 1402*3)
        
        '''
        self.mlps = []
        self.mlps.append(nn.Conv1d(self.k[-1], fc_n[0], 1))
        for i in range(len(fc_n)-1) :
            self.mlps.append(nn.Conv1d(fc_n[i], fc_n[i+1], 1))
        self.mlps = nn.ModuleList(self.mlps)
        
        self.bns = []
        for i in range(len(fc_n)) :
            self.bns.append(nn.BatchNorm1d(fc_n[i]))
        self.bns = nn.ModuleList(self.bns)
        self.last_mlp = nn.Conv1d(fc_n[-1], 3, 1)
        '''

    def forward(self, x, mesh, writer=False, steps=None):
        #print(len(mesh[0].vs))
        x_max, x_min = max(mesh[0].vs[:,0]), min(mesh[0].vs[:,0])
        y_max, y_min = max(mesh[0].vs[:,1]), min(mesh[0].vs[:,1])
        z_max, z_min = max(mesh[0].vs[:,2]), min(mesh[0].vs[:,2])
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(mesh[0].vs[:,0], mesh[0].vs[:,1], mesh[0].vs[:,2], marker='.')
        ax.set_xlim((x_min, x_max)); ax.set_ylim((y_min, y_max)); ax.set_zlim((z_min, z_max));
        # plt.savefig('/home/students/jlee/repos/meshcnn/original_mesh.png', bbox_inches='tight')
        if writer is not False:
            writer.add_figure('orignal/mesh', fig, steps)
            #vsfs = mesh[0].get_vs_fs()
            #writer.add_mesh('original/pointcloud', vertices=vsfs['vs'], global_step=steps)
            #writer.add_mesh('origianl/mesh', vertices=vsfs['vs'], faces=vsfs['fs'], global_step=steps)
        plt.close()
        
        for i in range(len(self.k) - 1):
            x = getattr(self, 'conv{}'.format(i))(x, mesh)
            x = F.relu(getattr(self, 'norm{}'.format(i))(x))
            x = getattr(self, 'pool{}'.format(i))(x, mesh)
            
            #print(len(vs))
            
            # plt.savefig('/home/students/jlee/repos/meshcnn/pooled_mesh_%d.png'%(i), bbox_inches='tight')
            if writer is not False :
                vs = mesh[0].vs[mesh[0].v_mask]
                fig = plt.figure()
                ax = plt.axes(projection='3d')
                ax.scatter3D(vs[:,0], vs[:,1], vs[:,2], marker='.')
                ax.set_xlim((x_min, x_max)); ax.set_ylim((y_min, y_max)); ax.set_zlim((z_min, z_max));
                writer.add_figure('pooled_mesh/%d'%(i+1), fig, steps)
                #vs, fs = mesh[0].get_vs_fs()
                #writer.add_mesh('pooled_pointcloud/%d'%(i+1), vertices=vsfs['vs'], global_step=steps)
                #writer.add_mesh('pooled_mesh/%d'%(i+1), vertices=vsfs['vs'], faces=vsfs['fs'], global_step=steps)
                plt.close()    

        x = self.gp(x)
        x = x.view(-1, self.k[-1])
        x = self.decoder_fc1(x)
        x = self.decoder_fc2(x)
        x = self.decoder_fc3(x)
        x = x.view(-1, 1402, 3)
        #print(x.shape)
        '''
        x = x.view(-1, self.k[-1], 1).repeat(1, 1, 1402)
        #print(x.shape)
        for i in range(len(self.fc_n)):
            x = F.relu(self.bns[i](self.mlps[i](x)))
        x = self.last_mlp(x)
        x = x.transpose(2,1).contiguous()
        #print('last layer:', x.shape)
        '''
        if writer is not False :
            vs = x[0].data.cpu().numpy()
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            ax.scatter3D(vs[:,0], vs[:,1], vs[:,2], marker='.')
            ax.set_xlim((x_min, x_max)); ax.set_ylim((y_min, y_max)); ax.set_zlim((z_min, z_max));
            writer.add_figure('reconstructed_mesh', fig, steps)
            #vs, fs = mesh[0].get_vs_fs()
            #writer.add_mesh('pooled_pointcloud/%d'%(i+1), vertices=vsfs['vs'], global_step=steps)
            #writer.add_mesh('pooled_mesh/%d'%(i+1), vertices=vsfs['vs'], faces=vsfs['fs'], global_step=steps)
            plt.close() 
        return x

class MResConv(nn.Module):
    def __init__(self, in_channels, out_channels, skips=1):
        super(MResConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skips = skips
        self.conv0 = MeshConv(self.in_channels, self.out_channels, bias=False)
        for i in range(self.skips):
            setattr(self, 'bn{}'.format(i + 1), nn.BatchNorm2d(self.out_channels))
            setattr(self, 'conv{}'.format(i + 1),
                    MeshConv(self.out_channels, self.out_channels, bias=False))

    def forward(self, x, mesh):
        x = self.conv0(x, mesh)
        x1 = x
        for i in range(self.skips):
            x = getattr(self, 'bn{}'.format(i + 1))(F.relu(x))
            x = getattr(self, 'conv{}'.format(i + 1))(x, mesh)
        x += x1
        x = F.relu(x)
        return x


class MeshEncoderDecoder(nn.Module):
    """Network for fully-convolutional tasks (segmentation)
    """
    def __init__(self, pools, down_convs, up_convs, blocks=0, transfer_data=True):
        super(MeshEncoderDecoder, self).__init__()
        self.transfer_data = transfer_data
        self.encoder = MeshEncoder(pools, down_convs, blocks=blocks)
        unrolls = pools[:-1].copy()
        unrolls.reverse()
        self.decoder = MeshDecoder(unrolls, up_convs, blocks=blocks, transfer_data=transfer_data)

    def forward(self, x, meshes, writer, steps) :
        mesh = meshes[0]
        x_max, x_min = max(mesh.vs[:,0]), min(mesh.vs[:,0])
        y_max, y_min = max(mesh.vs[:,1]), min(mesh.vs[:,1])
        z_max, z_min = max(mesh.vs[:,2]), min(mesh.vs[:,2])
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(mesh.vs[:,0], mesh.vs[:,1], mesh.vs[:,2], marker='.')
        ax.set_xlim((x_min, x_max)); ax.set_ylim((y_min, y_max)); ax.set_zlim((z_min, z_max));
        # plt.savefig('/home/students/jlee/repos/meshcnn/original_mesh.png', bbox_inches='tight')
        if writer is not False:
            writer.add_figure('orignal_mesh', fig, steps)
        plt.close()
        
        fe, before_pool = self.encoder((x, meshes), writer, steps, (x_max, x_min, y_max, y_min, z_max, z_min))
        fe = self.decoder((fe, meshes), before_pool, writer, steps, (x_max, x_min, y_max, y_min, z_max, z_min))
        return fe

    def __call__(self, x, meshes, writer=False, steps=None):
        return self.forward(x, meshes, writer, steps)

class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, blocks=0, pool=0):
        super(DownConv, self).__init__()
        self.bn = []
        self.pool = None
        self.conv1 = MeshConv(in_channels, out_channels)
        self.conv2 = []
        for _ in range(blocks):
            self.conv2.append(MeshConv(out_channels, out_channels))
            self.conv2 = nn.ModuleList(self.conv2)
        for _ in range(blocks + 1):
            self.bn.append(nn.InstanceNorm2d(out_channels))
            self.bn = nn.ModuleList(self.bn)
        if pool:
            self.pool = MeshPool(pool)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        fe, meshes = x
        x1 = self.conv1(fe, meshes)
        if self.bn:
            x1 = self.bn[0](x1)
        x1 = F.relu(x1)
        x2 = x1
        for idx, conv in enumerate(self.conv2):
            x2 = conv(x1, meshes)
            if self.bn:
                x2 = self.bn[idx + 1](x2)
            x2 = x2 + x1
            x2 = F.relu(x2)
            x1 = x2
        x2 = x2.squeeze(3)
        before_pool = None
        if self.pool:
            before_pool = x2
            x2 = self.pool(x2, meshes)
        return x2, before_pool


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, blocks=0, unroll=0, residual=True,
                 batch_norm=True, transfer_data=True):
        super(UpConv, self).__init__()
        self.residual = residual
        self.bn = []
        self.unroll = None
        self.transfer_data = transfer_data
        self.up_conv = MeshConv(in_channels, out_channels)
        if transfer_data:
            self.conv1 = MeshConv(2 * out_channels, out_channels)
        else:
            self.conv1 = MeshConv(out_channels, out_channels)
        self.conv2 = []
        for _ in range(blocks):
            self.conv2.append(MeshConv(out_channels, out_channels))
            self.conv2 = nn.ModuleList(self.conv2)
        if batch_norm:
            for _ in range(blocks + 1):
                self.bn.append(nn.InstanceNorm2d(out_channels))
            self.bn = nn.ModuleList(self.bn)
        if unroll:
            self.unroll = MeshUnpool(unroll)

    def __call__(self, x, from_down=None):
        return self.forward(x, from_down)

    def forward(self, x, from_down):
        from_up, meshes = x
        x1 = self.up_conv(from_up, meshes).squeeze(3)
        if self.unroll:
            x1 = self.unroll(x1, meshes)
        if self.transfer_data:
            x1 = torch.cat((x1, from_down), 1)
        x1 = self.conv1(x1, meshes)
        if self.bn:
            x1 = self.bn[0](x1)
        x1 = F.relu(x1)
        x2 = x1
        for idx, conv in enumerate(self.conv2):
            x2 = conv(x1, meshes)
            if self.bn:
                x2 = self.bn[idx + 1](x2)
            if self.residual:
                x2 = x2 + x1
            x2 = F.relu(x2)
            x1 = x2
        x2 = x2.squeeze(3)
        return x2


class MeshEncoder(nn.Module):
    def __init__(self, pools, convs, fcs=None, blocks=0, global_pool=None):
        super(MeshEncoder, self).__init__()
        self.fcs = None
        self.convs = []
        for i in range(len(convs) - 1):
            if i + 1 < len(pools):
                pool = pools[i + 1]
            else:
                pool = 0
            self.convs.append(DownConv(convs[i], convs[i + 1], blocks=blocks, pool=pool))
        self.global_pool = None
        if fcs is not None:
            self.fcs = []
            self.fcs_bn = []
            last_length = convs[-1]
            if global_pool is not None:
                if global_pool == 'max':
                    self.global_pool = nn.MaxPool1d(pools[-1])
                elif global_pool == 'avg':
                    self.global_pool = nn.AvgPool1d(pools[-1])
                else:
                    assert False, 'global_pool %s is not defined' % global_pool
            else:
                last_length *= pools[-1]
            if fcs[0] == last_length:
                fcs = fcs[1:]
            for length in fcs:
                self.fcs.append(nn.Linear(last_length, length))
                self.fcs_bn.append(nn.InstanceNorm1d(length))
                last_length = length
            self.fcs = nn.ModuleList(self.fcs)
            self.fcs_bn = nn.ModuleList(self.fcs_bn)
        self.convs = nn.ModuleList(self.convs)
        reset_params(self)

    def forward(self, x, writer=False, steps=None, lims=None):
        fe, meshes = x
        encoder_outs = []
        i = 0
        for conv in self.convs:
            fe, before_pool = conv((fe, meshes))
            encoder_outs.append(before_pool)
            
            # FOR TENSORBOARD
            if writer is not False:
                mesh = meshes[0]
                x_max, x_min, y_max, y_min, z_max, z_min = lims
                vs = mesh.vs[mesh.v_mask]
                #print(len(vs))
                fig = plt.figure()
                ax = plt.axes(projection='3d')
                ax.scatter3D(vs[:,0], vs[:,1], vs[:,2], marker='.')
                ax.set_xlim((x_min, x_max)); ax.set_ylim((y_min, y_max)); ax.set_zlim((z_min, z_max));
                # plt.savefig('/home/students/jlee/repos/meshcnn/original_mesh.png', bbox_inches='tight')
                writer.add_figure('pooled_mesh/%d'%(i), fig, steps)
                plt.close()
            i += 1   
        if self.fcs is not None:
            if self.global_pool is not None:
                fe = self.global_pool(fe)
            fe = fe.contiguous().view(fe.size()[0], -1)
            for i in range(len(self.fcs)):
                fe = self.fcs[i](fe)
                if self.fcs_bn:
                    x = fe.unsqueeze(1)
                    fe = self.fcs_bn[i](x).squeeze(1)
                if i < len(self.fcs) - 1:
                    fe = F.relu(fe)
        return fe, encoder_outs

    def __call__(self, x, writer=False, steps=None, lims=None):
        return self.forward(x, writer, steps, lims)


class MeshDecoder(nn.Module):
    def __init__(self, unrolls, convs, blocks=0, batch_norm=True, transfer_data=True):
        super(MeshDecoder, self).__init__()
        self.up_convs = []
        for i in range(len(convs) - 2):
            if i < len(unrolls):
                unroll = unrolls[i]
            else:
                unroll = 0
            self.up_convs.append(UpConv(convs[i], convs[i + 1], blocks=blocks, unroll=unroll,
                                        batch_norm=batch_norm, transfer_data=transfer_data))
        self.final_conv = UpConv(convs[-2], convs[-1], blocks=blocks, unroll=False,
                                 batch_norm=batch_norm, transfer_data=False)
        self.up_convs = nn.ModuleList(self.up_convs)
        reset_params(self)

    def forward(self, x, encoder_outs=None, writer=False, steps=None, lims=None):
        fe, meshes = x
        for i, up_conv in enumerate(self.up_convs):
            before_pool = None
            if encoder_outs is not None:
                before_pool = encoder_outs[-(i+2)]
            fe = up_conv((fe, meshes), before_pool)
            
            # FOR TENSORBOARD
            if writer is not False:
                mesh = meshes[0]
                x_max, x_min, y_max, y_min, z_max, z_min = lims
                vs = mesh.vs[mesh.v_mask]
                #print(len(vs))
                fig = plt.figure()
                ax = plt.axes(projection='3d')
                ax.scatter3D(vs[:,0], vs[:,1], vs[:,2], marker='.')
                ax.set_xlim((x_min, x_max)); ax.set_ylim((y_min, y_max)); ax.set_zlim((z_min, z_max));
                # plt.savefig('/home/students/jlee/repos/meshcnn/original_mesh.png', bbox_inches='tight')
                writer.add_figure('unpooled_mesh/%d'%(i), fig, steps)
                plt.close()
                
        fe = self.final_conv((fe, meshes))
        # FOR TENSORBOARD
        if writer is not False:
            mesh = meshes[0]
            x_max, x_min, y_max, y_min, z_max, z_min = lims
            vs = mesh.vs[mesh.v_mask]
            #print(len(vs))
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            ax.scatter3D(vs[:,0], vs[:,1], vs[:,2], marker='.')
            ax.set_xlim((x_min, x_max)); ax.set_ylim((y_min, y_max)); ax.set_zlim((z_min, z_max));
            # plt.savefig('/home/students/jlee/repos/meshcnn/original_mesh.png', bbox_inches='tight')
            writer.add_figure('unpooled_mesh/%d'%(i+1), fig, steps)
            plt.close()
        return fe

    def __call__(self, x, encoder_outs=None, writer=False, steps=None, lims=None):
        return self.forward(x, encoder_outs, writer, steps, lims)

def reset_params(model): # todo replace with my init
    for i, m in enumerate(model.modules()):
        weight_init(m)

def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
