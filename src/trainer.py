from tqdm import tqdm
import torch.optim as optim
import torch
from tensorboardX import SummaryWriter
import os
import numpy as np

from utils import set_color


class Trainer(object):

    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay

        self.optimizer = self._build_optimizer(name=args.optimizer, params=self.model.parameters())
        self.loss_func = torch.nn.BCELoss()
        self.logistic = torch.nn.Sigmoid()

        self._writer = SummaryWriter(log_dir=args.tensorboard_dir)
        # self.saved_model_file = os.path.join(args.checkpoint_dir, '/{}.pth'.format(args.dataset))

    def _build_optimizer(self, name, params):
        r"""Init the Optimizer

        Returns:
            torch.optim: the optimizer
        """
        if name.lower() == 'adam':
            optimizer = optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif name.lower() == 'sgd':
            optimizer = optim.SGD(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif name.lower() == 'adagrad':
            optimizer = optim.Adagrad(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif name.lower() == 'rmsprop':
            optimizer = optim.RMSprop(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            print('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = optim.Adam(params, lr=self.learning_rate)
        return optimizer

    def train_an_epoch(self, train_data, epoch_id):
        self.model.train()
        total_loss = 0
        iter_data = tqdm(train_data, total=len(train_data), ncols=100, desc=set_color(f"Train {epoch_id:>5}", 'pink'))
        for batch_id, interaction in enumerate(iter_data):
            self.optimizer.zero_grad()
            loss = self.model.calculate_loss(interaction, self.loss_func, self.logistic)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            iter_data.set_postfix(Batch_Loss=loss.item())
        self._writer.add_scalar('model/loss', total_loss, epoch_id)

    def evaluate(self, test_data, ground_true_items, mask_matrix, epoch_id):
        pred_list = []  # predicted item list
        ground_true_list = []  # ture item list
        self.model.eval()
        with torch.no_grad():
            iter_data = tqdm(test_data, total=len(test_data), ncols=100, desc=set_color(f"Evaluate   ", 'pink'))
            for batch_idx, batch_users in enumerate(iter_data):
                batch_users = batch_users.to(self.args.device)
                scores = self.model.predict(batch_users).cpu()  # batch_user * n_items
                scores += mask_matrix[batch_users]  # mask history interaction

                _, pred = torch.topk(scores, k=self.args.topk)
                pred_list.append(pred.numpy().tolist())  # list shape: batch_num* batch_user * k
                ground_true_list.append([ground_true_items[int(u)] for u in batch_users])  # list

        X = zip(pred_list, ground_true_list)
        Recall, Precision, NDCG = 0, 0, 0
        for i, x in enumerate(X):
            precision, recall, ndcg = self._evaluate_one_batch(x)
            Recall += recall
            Precision += precision
            NDCG += ndcg

        n_user = self.model.n_users
        # Precision /= n_user
        # Recall /= n_user
        NDCG /= n_user
        # F1_score = 2 * (Precision * Recall) / (Precision + Recall)
        self._writer.add_scalar('model/NDCG', NDCG, epoch_id)
        print("NDCG: {:.5f}".format(NDCG))

    def _evaluate_one_batch(self, x):
        pred_items = x[0]  # list: batch_user * k
        ground_true_items = x[1]  # list: batch_user * n (n is the num of ground true items)
        hit = []
        for i in range(len(pred_items)):
            ground_true = ground_true_items[i]
            pred = pred_items[i]
            pred_in_groundtrue = list(map(lambda x: x in ground_true, pred))  # [True, False, ...] len: topk
            pred_in_groundtrue = np.array(pred_in_groundtrue).astype('float')
            hit.append(pred_in_groundtrue)
        hit = np.array(hit).astype('float')  # np.array: batch_user * k
        precision = 0
        recall = 0
        ndcg = self._NDCG_AT_K(ground_true_items, hit)

        return precision, recall, ndcg

    def _NDCG_AT_K(self, ground_true_items, hit):
        assert len(ground_true_items) == len(hit)
        batch_users_num = len(hit)
        k = self.args.topk

        # calculate dcg
        dcg = hit * (1.0 / np.log2(np.arange(2, k + 2)))
        dcg = np.sum(dcg, axis=1)

        # calculate idcg
        idcg_matrix = np.zeros((batch_users_num, k))
        for i, items in enumerate(ground_true_items):
            length = k if k <= len(items) else len(items)
            idcg_matrix[i, :length] = 1
        idcg = idcg_matrix * (1.0 / np.log2(np.arange(2, k + 2)))
        idcg = np.sum(idcg, axis=1)

        # calculate ndcg
        ndcg = dcg / idcg
        return np.sum(ndcg)
