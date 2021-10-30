import math
import torch
from collections import OrderedDict
from torch.nn import utils, functional as F
from torch.optim import Adam
from torch.backends import cudnn
from torchvision import transforms
from dssnet import build_model, weights_init
from loss import Loss
import numpy as np
import PIL
import os 
import matplotlib.pyplot as plt

# from tools.visual import Viz_visdom

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()

class Solver(object):
    def __init__(self, train_loader, val_loader, test_dataset, config):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_dataset = test_dataset
        self.config = config
        self.beta = math.sqrt(0.3)  # for max F_beta metric

        # inference: choose the side map (see paper)
        self.select = [1, 2, 3, 6]
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.Tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        if self.config.cuda:
            cudnn.benchmark = True
            self.device = torch.device('cuda:0')


        self.build_model()
        if self.config.pre_trained: self.net.load_state_dict(torch.load(self.config.pre_trained))
        if config.mode == 'train':
            self.log_output = open("%slogs/log.txt" % config.save_fold, 'w')
        else:
            print(config.save_fold)
            fname = '%s/run-5/models/epoch_%d.pth' % (self.config.save_fold, 375)
            current_epoch, global_counter = self.load_ckp(fname)
            # self.net.load_state_dict(torch.load(self.config.model))
            self.net.eval()
            self.test_output = open("%s/test.txt" % config.test_fold, 'w')
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    # print the network information and parameter numbers
    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            if p.requires_grad: num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    def save_checkpoint(self, state, epoch):     # save points during training
        """Function for saving checkpoints"""
        savepath = '%s/models/epoch_%d.pth' % (self.config.save_fold, epoch + 1)
        torch.save(state, savepath)
        print(f"Model saved to {savepath}")

    def load_ckp(self, checkpoint_fpath):
        checkpoint = torch.load(checkpoint_fpath)
        self.net.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        return checkpoint['epoch']+1, checkpoint['global_counter']


    # build the network
    def build_model(self):
        self.net = build_model().to(self.device)
        if self.config.mode == 'train': self.loss = Loss().to(self.device)
        self.net.train()
        self.net.apply(weights_init)
        if self.config.load == '':
            self.net.base.load_state_dict(torch.load(self.config.vgg))
        if self.config.load != '':
            self.net.load_state_dict(torch.load(self.config.load))
        self.optimizer = Adam(self.net.parameters(), self.config.lr)
        # self.print_network(self.net, 'DSS')

    # update the learning rate
    def update_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    # evaluate MAE (for test or validation phase)
    def eval_mae(self, y_pred, y):
        return torch.abs(y_pred - y).mean()

    # TODO: write a more efficient version
    # get precisions and recalls: threshold---divided [0, 1] to num values
    def eval_pr(self, y_pred, y, num):
        prec, recall = torch.zeros(num), torch.zeros(num)
        thlist = torch.linspace(0, 1 - 1e-10, num)
        for i in range(num):
            y_temp = (y_pred >= thlist[i]).float()
            tp = (y_temp * y).sum()
            prec[i], recall[i] = tp / (y_temp.sum() + 1e-20), tp / y.sum()
        return prec, recall

    # validation: using resize image, and only evaluate the MAE metric
    def validation(self):
        avg_mae = 0.0
        self.net.eval()
        with torch.no_grad():
            for i, data_batch in enumerate(self.val_loader):
                images, labels = data_batch
                images, labels = images.to(self.device), labels.to(self.device)
                prob_pred = self.net(images)
                prob_pred = torch.mean(torch.cat([prob_pred[i] for i in self.select], dim=1), dim=1, keepdim=True)
                avg_mae += self.eval_mae(prob_pred, labels).item()
        self.net.train()
        return avg_mae / len(self.val_loader)

    # test phase: using origin image size, evaluate MAE and max F_beta metrics
    def test(self, num, use_crf=False):
        if use_crf: 
            from tools.crf_process import crf
        avg_mae, img_num = 0.0, len(self.test_dataset)
        avg_prec, avg_recall = torch.zeros(num), torch.zeros(num)
        with torch.no_grad():
            for i, (img, labels, names) in enumerate(self.test_dataset):
                # print(names)
                images = self.transform(img).unsqueeze(0)
                labels = labels.unsqueeze(0)
                shape = labels.size()[2:]
                images = images.to(self.device)
                prob_pred = self.net(images)
                
                save_dir  = '/home/charis/kul-thesis/SGAN/DSS-pytorch/results/saliencies/'
                im_name = os.path.join(save_dir, names + '.npy')
                im_name_png = os.path.join(save_dir, names[0:-4] + '.png')
                
                prob_pred = torch.mean(torch.cat([prob_pred[i] for i in self.select], dim=1), dim=1, keepdim=True)
                prob_pred = F.interpolate(prob_pred, size=shape, mode='bilinear', align_corners=True).cpu().data
                if use_crf:
                    prob_pred = crf(img, prob_pred.numpy(), to_tensor=True)
                # print(prob_pred.shape)
                prob_pred2 = prob_pred.cpu().numpy()
                prob_pred2 = np.squeeze(np.squeeze(prob_pred2, axis=0), axis=0)
                # print(prob_pred2.shape)
                
                norm = (prob_pred2-prob_pred2.min()-1e-5) / (prob_pred2.max()  - prob_pred2.min() + 1e-5)
                final_img = np.array(norm * 255, dtype=np.uint8)                                         # make to image
                final_img = PIL.Image.fromarray(final_img)
                plt.imsave(im_name_png, final_img, cmap='gray')
                # np.save(im_name, prob_pred)
                
                mae = self.eval_mae(prob_pred, labels)
                prec, recall = self.eval_pr(prob_pred, labels, num)
                print("[%d] mae: %.4f" % (i, mae))
                print("[%d] mae: %.4f" % (i, mae), file=self.test_output)
                avg_mae += mae
                avg_prec, avg_recall = avg_prec + prec, avg_recall + recall
        avg_mae, avg_prec, avg_recall = avg_mae / img_num, avg_prec / img_num, avg_recall / img_num
        score = (1 + self.beta ** 2) * avg_prec * avg_recall / (self.beta ** 2 * avg_prec + avg_recall)
        score[score != score] = 0  # delete the nan
        # print('average mae: %.4f, max fmeasure: %.4f' % (avg_mae, score.max()))
        print('average mae: %.4f, max fmeasure: %.4f' % (avg_mae, score.max()), file=self.test_output)

    # training phase
    def train(self):
        iter_num = len(self.train_loader.dataset) // self.config.batch_size
        best_mae = 1.0 if self.config.val else None
        current_epoch = 0
        global_counter = 0
        if False:
            # Load from checkpoint
            fname = '%smodels/epoch_%d.pth' % (self.config.save_fold, 33)
            current_epoch, global_counter = self.load_ckp(fname)
            print(f'Weights loaded with current_epoch: {current_epoch} and global counter: {global_counter}!')
        while current_epoch < self.config.epoch:
            loss_epoch = 0
            for i, data_batch in enumerate(self.train_loader):
                if (i + 1) > iter_num: break
                self.net.zero_grad()
                x, y = data_batch
                x, y = x.to(self.device), y.to(self.device)
                y_pred = self.net(x)
                loss = self.loss(y_pred, y)
                loss.backward()
                utils.clip_grad_norm_(self.net.parameters(), self.config.clip_gradient)
                # utils.clip_grad_norm(self.loss.parameters(), self.config.clip_gradient)
                self.optimizer.step()
                global_counter += 1
                loss_epoch += loss.item()
            print('epoch: [%d/%d], iter: [%d/%d], loss: [%.4f]' % (
                current_epoch, self.config.epoch, i, iter_num, loss.item()))


            if (current_epoch + 1) % self.config.epoch_show == 0:
                print('epoch: [%d/%d], epoch_loss: [%.4f]' % (current_epoch, self.config.epoch, loss_epoch / iter_num),
                      file=self.log_output)

            mae = self.validation()
            print('--- Best MAE: %.2f, Curr MAE: %.2f ---' % (best_mae, mae))
            print('--- Best MAE: %.2f, Curr MAE: %.2f ---' % (best_mae, mae), file=self.log_output)
            if best_mae > mae:
                best_mae = mae
                self.save_checkpoint(
                      {'epoch': current_epoch,
                        'global_counter': global_counter,
                        'state_dict':self.net.state_dict(),
                        'optimizer':self.optimizer.state_dict()
                      }, current_epoch)
            if (current_epoch + 1) % self.config.epoch_save == 0:
                self.save_checkpoint(
                      {'epoch': current_epoch,
                        'global_counter': global_counter,
                        'state_dict':self.net.state_dict(),
                        'optimizer':self.optimizer.state_dict()
                      }, current_epoch)
            current_epoch += 1
        # self.save_checkpoint(
        #               {'epoch': current_epoch,
        #                 'global_counter': global_counter,
        #                 'state_dict':self.net.state_dict(),
        #                 'optimizer':self.optimizer.state_dict()
        #               }, current_epoch)