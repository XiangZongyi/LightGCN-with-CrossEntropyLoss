import torch
import numpy as np
import torch.nn as nn
from torch.nn.init import xavier_normal_, xavier_uniform_, constant_
import scipy.sparse as sp


class LightGCNPseudo(nn.Module):
    def __init__(self, args, dataset):
        super(LightGCNPseudo, self).__init__()

        self.args = args

        # load dataset info
        self.n_users = dataset.user_num
        self.n_items = dataset.item_num
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)

        # load parameters
        self.embedding_size = args.embedding_size
        self.n_layers = args.n_layers
        self.weight_decay = args.weight_decay

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.loss = nn.BCEWithLogitsLoss()  # Sigmoid + BCELoss

        # get adjacent matrix
        self.norm_adj_matrix = self._get_norm_adj_mat()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # parameters initialization
        self._init_weights()
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']

    def _init_weights(self):
        xavier_uniform_(self.user_embedding.weight.data)
        xavier_uniform_(self.item_embedding.weight.data)

    def _get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # build adj matrix
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)

        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)

        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D

        # covert norm_adj matrix to tensor  time4: 0.0040
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = np.array([row, col])
        i = torch.LongTensor(i)
        # i = torch.LongTensor([row, col])
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL.to(self.args.device)

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings

    def calculate_loss(self, interaction, loss_func, logistic):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user, item, rating = interaction[0], interaction[1], interaction[2]
        user = user.to(self.args.device)
        item = item.to(self.args.device)
        rating = rating.to(self.args.device)

        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]

        # calculate BCE Loss
        logits = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        pred_rating = logistic(logits)

        # calculate BCE Loss
        loss = loss_func(pred_rating, rating)

        return loss

    def predict(self, user):
        """
        :param user: the id of batch users
        :return:
        """
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores
