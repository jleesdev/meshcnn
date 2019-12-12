import torch
from . import networks
from os.path import join
from util.util import seg_accuracy, print_network
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import sys 
sys.path.append('/home/students/jlee/libs/chamferdist')
from chamferdist import ChamferDistance
sys.path.append('/home/students/jlee/libs/icp')
from icp import icp_R, icp
import numpy as np
from emd import EMDLoss
from pypoisson import poisson_reconstruction
import point_cloud_utils as pcu
import scipy
import os

class AutoEncoderModel:
    """ Class for training Model weights

    :args opt: structure containing configuration params
    e.g.,
    --dataset_mode -> classification / segmentation)
    --arch -> network type
    """
    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.is_train = opt.is_train
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.save_dir = join(opt.checkpoints_dir, opt.name)
        self.optimizer = None
        self.edge_features = None
        self.labels = None
        self.mesh = None
        self.soft_label = None
        self.loss = None

        #
        self.nclasses = opt.nclasses

        # load/define networks
        self.net = networks.define_classifier(opt.input_nc, opt.ncf, opt.ninput_edges, opt.nclasses, opt,
                                              self.gpu_ids, opt.arch, opt.init_type, opt.init_gain)
        self.net.train(self.is_train)
        self.criterion = networks.define_loss(opt).to(self.device)

        if self.is_train:
            if opt.optim == 'RMSprop' :
                self.optimizer = torch.optim.RMSprop(self.net.parameters(), lr=opt.lr)
            else:
                self.optimizer = torch.optim.Adam(self.net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.scheduler = networks.get_scheduler(self.optimizer, opt)
            print_network(self.net)

        if not self.is_train or opt.continue_train:
            self.load_network(opt.which_epoch)

    def set_input(self, data):
        input_edge_features = torch.from_numpy(data['edge_features']).float()
        #print(data['label'].dtype, data['label'].shape)
        labels = torch.from_numpy(data['label']).float()
        # set inputs
        self.edge_features = input_edge_features.to(self.device).requires_grad_(self.is_train)
        self.labels = labels.to(self.device)
        self.init_faces = data['init_faces']
        self.pad_iter = data['pad_iter']
        self.mesh = data['mesh']
        self.export_folder = data['export_folder']
        self.filename = data['filename']
        if self.opt.dataset_mode == 'segmentation' and not self.is_train:
            self.soft_label = torch.from_numpy(data['soft_label'])


    def forward(self, writer=False, steps=None):
        out = self.net(self.edge_features, self.mesh, writer, steps)
        # print(out.shape)
        return out

    def backward(self, out):
        dist1, dist2, idx1, idx2= self.criterion(self.labels, out)
        self.loss = 0.5 * (dist1.mean() + dist2.mean())
        self.loss.backward()

    def optimize_parameters(self, writer=False, steps=None):
        self.optimizer.zero_grad()
        out = self.forward(writer, steps)
        #print(self.labels)
        self.backward(out)
        self.optimizer.step()


##################

    def load_network(self, which_epoch):
        """load model from disk"""
        save_filename = '%s_net.pth' % which_epoch
        load_path = join(self.save_dir, save_filename)
        net = self.net
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        print('loading the model from %s' % load_path)
        # PyTorch newer than 0.4 (e.g., built from
        # GitHub source), you can remove str() on self.device
        state_dict = torch.load(load_path, map_location=str(self.device))
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        net.load_state_dict(state_dict)


    def save_network(self, which_epoch):
        """save model to disk"""
        save_filename = '%s.pth' % (which_epoch)
        save_path = join(self.save_dir, save_filename)
        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            torch.save(self.net.module.cpu().state_dict(), save_path)
            self.net.cuda(self.gpu_ids[0])
        else:
            torch.save(self.net.cpu().state_dict(), save_path)

    def update_learning_rate(self):
        """update learning rate (called once every epoch)"""
        self.scheduler.step()
        lr = self.optimizer.param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def test(self):
        """tests model
        returns: number correct and total number
        """
        with torch.no_grad():
            out = self.forward()
            dist1, dist2, _, _ = self.criterion(self.labels, out)
            test_loss = 0.5 * (dist1.mean() + dist2.mean())
            
            labels = torch.Tensor.cpu(self.labels).numpy()
            out = torch.Tensor.cpu(out).numpy()
            
            for i in range(out.shape[0]) :
                normals = pcu.estimate_normals(out[i], k=16)
                #print(n.shape, n)
                fs, vs = poisson_reconstruction(out[i], normals, depth=5, full_depth=3)
                print(vs.shape, fs.shape)
                filename, file_extension = os.path.splitext(self.filename[i])
                file = '%s/%s_%s%s' % (self.export_folder[i], filename, 'out', file_extension)
                print(file)
                self.output_export(vs, fs, file)
            
            #print(labels.shape, out.shape)
            #for i in range(len(labels)) :
            #    mean_error, R, indices = icp(labels[i], out[i], tolerance=0.0001)
            #    print(out[i].shape, out[i])
            #    print(indices.shape, np.unique(indices).shape, indices, np.unique(indices))
            #print(out[indices[:len(indices)-self.pad_iter]].shape, self.init_faces)
            #self.output_export(vertices, faces)
            print(test_loss, len(self.labels))
        return test_loss, len(self.labels), out, self.labels

    def get_accuracy(self, pred, labels):
        """computes accuracy for classification / segmentation """
        if self.opt.dataset_mode == 'classification':
            correct = pred.eq(labels).sum()
        elif self.opt.dataset_mode == 'segmentation':
            correct = seg_accuracy(pred, self.soft_label, self.mesh)
        elif self.opt.dataset_mode == 'autoencoder' :
            correct = torch.nn.functional.l1_loss(pred, labels)
        return correct

    def output_export(self, out, faces, file=None, vcolor=None):
        if file is None:
            if self.export_folder :
                filename, file_extension = os.path.splitext(self.filename)
                file = '%s/%s_%s%s' % (self.export_folder, filename, 'out', file_extension)
            else:
                return
        faces = faces
        vs = out
        
        with open(file, 'w+') as f:
            for vi, v in enumerate(vs):
                vcol = ' %f %f %f' % (vcolor[vi, 0], vcolor[vi, 1], vcolor[vi, 2]) if vcolor is not None else ''
                f.write("v %f %f %f%s\n" % (v[0], v[1], v[2], vcol))
            for face_id in range(len(faces) - 1):
                f.write("f %d %d %d\n" % (faces[face_id][0] + 1, faces[face_id][1] + 1, faces[face_id][2] + 1))
            f.write("f %d %d %d" % (faces[-1][0] + 1, faces[-1][1] + 1, faces[-1][2] + 1))
            