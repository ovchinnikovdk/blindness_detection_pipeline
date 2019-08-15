import torch
import tqdm
from lib.loader import LoaderFactory
from lib.postprocessing import PostProcessor
import numpy as np
import os


class NetTrainer(object):
    def __init__(self, net, gpu, batch_size, epochs, data_path):
        self.net = net
        self.metrics = {'accuracy': lambda x, y: 1.}
        self.gpu = gpu
        self.batch_size = batch_size
        self.epochs = epochs
        self.data_path = data_path
        self.train_loader, self.val_loader = LoaderFactory(self.data_path).create_loaders()
        self.postprocessor = PostProcessor()
        self.model_version = 'dummy1'
        self.save_path = '/logs'
        self.loss = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max')
        self.score_history = [0.]
        self.current_epoch = 0

    def train(self):
        train_loader = self.train_loader
        if self.gpu:
            self.net.cuda()
        else:
            self.net.cpu()
        self.net.train()
        for i in range(self.epochs):
            print("EPOCH #" + str(i) + ' of ' + str(self.epochs))
            self.net.train()
            sum_loss = 0
            for x, y in tqdm.tqdm(train_loader, desc='Training epoch #' + str(i)):
                if self.gpu:
                    x = x.cuda()
                    y = y.cuda()
                self.optimizer.zero_grad()
                output = self.net(x)
                loss_out = self.loss(output, y)
                loss_out.backward()
                self.optimizer.step()
                sum_loss += loss_out.item()
            print("Loss: " + str(sum_loss))
            self.current_epoch = i
            self._validate()

    def _validate(self):
        # Validating Epoch
        torch.cuda.empty_cache()
        self.net.eval()
        pred_y = []
        true_y = []
        val_score = dict()
        with torch.no_grad():
            val_loss = 0.
            for val_x, val_y in tqdm.tqdm(self.val_loader, desc='Validating epoch #' + str(self.current_epoch)):
                if self.gpu:
                    val_x = val_x.cuda()
                    val_y = val_y.cuda()
                pred = self.net(val_x)
                loss_out = self.loss(pred, val_y)
                val_loss += loss_out.item()
                pred_y.append(self.postprocessor.process(pred.cpu()))
                true_y.append(val_y.cpu())
                torch.cuda.empty_cache()
            pred_y = torch.cat(pred_y, dim=0)
            true_y = torch.cat(true_y, dim=0)
            print("Validation loss: {0:10.5f}".format(val_loss))
            for metric in self.metrics.keys():
                val_score[metric] = self.metrics[metric](pred_y, true_y)
            print(val_score)
            val_score_mean = np.mean(list(val_score.values()))
            if val_score_mean > max(self.score_history):
                if not os.path.exists(self.save_path):
                    os.mkdir(self.save_path)
                torch.save(self.net.state_dict(), os.path.join(self.save_path, self.model_version +
                                                               str(self.current_epoch) + '.dat'))
            self.score_history.append(val_score_mean)
            self.scheduler.step(val_score_mean)
