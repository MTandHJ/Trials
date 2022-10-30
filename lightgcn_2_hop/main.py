

from typing import Dict, Optional, Union

import torch, os
import torch_geometric.transforms as T
import numpy as np
import scipy.sparse as sp
from torch_geometric.data import Data, HeteroData 
from torch_geometric.nn import LGConv, MessagePassing, GCNConv, GCN
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import to_scipy_sparse_matrix

from freeplot.utils import import_pickle, export_pickle
from freerec.parser import Parser
from freerec.launcher import Coach
from freerec.models import RecSysArch
from freerec.criterions import BPRLoss
from freerec.data.datasets import Gowalla_m1, Yelp18_m1, AmazonBooks_m1
from freerec.data.fields import Tokenizer
from freerec.data.tags import USER, ITEM, ID
from freerec.utils import timemeter, mkdirs


cfg = Parser()
cfg.add_argument("-eb", "--embedding-dim", type=int, default=64)
cfg.add_argument("--layers", type=int, default=3)
cfg.add_argument("--num-filters", type=int, default=100)
cfg.add_argument("--filter_type", type=str, default='none')
cfg.add_argument("--alpha", type=float, default=1.)
cfg.add_argument("--require-linear", type=bool, default=True)
cfg.add_argument("--require-activation", type=bool, default=False)
cfg.set_defaults(
    description="LightGCN",
    root="../../data",
    dataset='Gowalla_m1',
    epochs=500,
    batch_size=2048,
    optimizer='adam',
    lr=1e-3,
    weight_decay=1e-8,
    seed=2020
)
cfg.compile()

class LightGCN(RecSysArch):

    def __init__(
        self, tokenizer: Tokenizer, 
        graph: Data,
    ) -> None:
        super().__init__()

        self.tokenizer = tokenizer
        self.num_layers = cfg.layers
        self.User, self.Item = self.tokenizer[USER, ID], self.tokenizer[ITEM, ID]

        self.graph = graph

        self.dense = torch.nn.ModuleList()
        self.dense.append(torch.nn.Linear(cfg.num_filters, cfg.embedding_dim, bias=False))
        self.dense.append(torch.nn.Linear(cfg.num_filters, cfg.embedding_dim, bias=False))

        self.initialize()

        if cfg.require_linear:
            for _ in range(cfg.layers):
                self.dense.append(
                    GCNConv(
                        cfg.embedding_dim, cfg.embedding_dim, normalize=False
                    )
                )
        
        else:
            for _ in range(cfg.layers):
                self.dense.append(LGConv(normalize=False))

        if cfg.require_activation:
            self.act = torch.nn.ReLU()
        else:
            self.act = torch.nn.Identity()


    @property
    def graph(self):
        return self.__graph

    @graph.setter
    def graph(self, graph: Data):
        self.__graph = graph
        T.ToSparseTensor()(self.__graph)
        self.__graph.adj_t = gcn_norm(
            self.__graph.adj_t, num_nodes=self.User.count + self.Item.count,
            add_self_loops=False
        )
        self.__graph.adj_t = self.__graph.adj_t.matmul(self.__graph.adj_t.t())

    def to(
        self, device: Optional[Union[int, torch.device]] = None, 
        dtype: Optional[Union[torch.dtype, str]] = None, 
        non_blocking: bool = False
    ):
        if device:
            self.graph.to(device)
        return super().to(device, dtype, non_blocking)

    def save(self, data: Dict):
        path = os.path.join("filters", cfg.dataset)
        mkdirs(path)
        file_ = os.path.join(path, "eig_vals_vecs.pickle")
        export_pickle(data, file_)

    @timemeter("LightGCN/load")
    def load(self, graph: HeteroData):
        path = os.path.join("filters", cfg.dataset)
        file_ = os.path.join(path, "eig_vals_vecs.pickle")
        try:
            data = import_pickle(file_)
        except ImportError:
            data = self.preprocess(graph)
        
        U, vals, V = data['U'], data['vals'], data['V']
        vals = self.weight_filter(vals[:cfg.num_filters])
        U = U[:, :cfg.num_filters] * vals
        V = V[:, :cfg.num_filters] * vals
        self.register_buffer("U", U)
        self.register_buffer("V", V)

    def weight_filter(self, vals: torch.Tensor):
        if cfg.filter_type == 'none':
            return vals
        elif cfg.filter_type == 'ideal':
            return np.ones_like(vals)
        else:
            return torch.exp(5. * vals)

    def preprocess(self, graph: HeteroData):
        R = sp.lil_array(to_scipy_sparse_matrix(
            graph[graph.edge_types[0]].edge_index,
            num_nodes=max(self.User.count, self.Item.count)
        ))[:self.User.count, :self.Item.count] # N x M
        userDegs = R.sum(axis=1).squeeze() + cfg.alpha
        itemDegs = R.sum(axis=0).squeeze() + cfg.alpha
        userDegs = 1 / np.sqrt(userDegs)
        itemDegs = 1 / np.sqrt(itemDegs)
        userDegs[np.isinf(userDegs)] = 0.
        itemDegs[np.isinf(itemDegs)] = 0.
        R = (userDegs.reshape(-1, 1) * R * itemDegs).tocoo()
        rows = torch.from_numpy(R.row).long()
        cols = torch.from_numpy(R.col).long()
        vals = torch.from_numpy(R.data)
        indices = torch.stack((rows, cols), dim=0)
        R = torch.sparse_coo_tensor(
            indices, vals, size=R.shape
        )

        U, vals, V = torch.svd_lowrank(R, q=400, niter=30)

        data = {'U': U.cpu(), 'vals': vals.cpu(), 'V': V.cpu()}
        self.save(data)
        return data

    def forward(
        self, users: Optional[Dict[str, torch.Tensor]] = None, 
        items: Optional[Dict[str, torch.Tensor]] = None
    ):
        userEmbs = self.dense[0](self.U)
        itemEmbs = self.dense[1](self.V)
        features = torch.cat((userEmbs, itemEmbs), dim=0).flatten(1) # N x D
        avgFeats = features / (self.num_layers + 1)
        for conv in self.dense[2:]:
            features = self.act(conv(features, self.graph.adj_t))
            avgFeats += features / (self.num_layers + 1)
        userFeats, itemFeats = torch.split(avgFeats, (self.User.count, self.Item.count))

        if self.training: # Batch
            users, items = users[self.User.name], items[self.Item.name]
            userFeats = userFeats[users] # B x 1 x D
            itemFeats = itemFeats[items] # B x n x D
            return torch.mul(userFeats, itemFeats).sum(-1)
        else:
            return userFeats, itemFeats



class CoachForLightGCN(Coach):


    def train_per_epoch(self):
        for users, items in self.dataloader:
            users = {name: val.to(self.device) for name, val in users.items()}
            items = {name: val.to(self.device) for name, val in items.items()}

            scores = self.model(users, items)
            pos, neg = scores[:, 0], scores[:, 1]
            loss = self.criterion(pos, neg)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.monitor(loss.item(), n=scores.size(0), mode="mean", prefix='train', pool=['LOSS'])

    def evaluate(self, prefix: str = 'valid'):
        User = self.fields[USER, ID]
        Item = self.fields[ITEM, ID]
        userFeats, itemFeats = self.model()
        for users, items in self.dataloader:
            users = users[User.name].to(self.device)
            targets = items[Item.name].to(self.device)
            users = userFeats[users].flatten(1) # B x D
            items = itemFeats.flatten(1) # N x D
            preds = users @ items.T # B x N
            preds[targets == -1] = -1e10
            targets[targets == -1] = 0

            self.monitor(
                preds, targets,
                n=len(users), mode="mean", prefix=prefix,
                pool=['NDCG', 'PRECISION', 'RECALL', 'HITRATE']
            )


def main():

    if cfg.dataset == "Gowalla_m1":
        basepipe = Gowalla_m1(cfg.root)
    elif cfg.dataset == "Yelp18_m1":
        basepipe = Yelp18_m1(cfg.root)
    elif cfg.dataset == "AmazonBooks_m1":
        basepipe = AmazonBooks_m1(cfg.root)
    else:
        raise ValueError("Dataset should be Gowalla_m1, Yelp18_m1 or AmazonBooks_m1")
    trainpipe = basepipe.shard_().uniform_sampling_(num_negatives=1).tensor_().split_(cfg.batch_size)
    validpipe = basepipe.trisample_(batch_size=cfg.batch_size).shard_().tensor_()
    dataset = trainpipe.wrap_(validpipe).group_((USER, ITEM))

    tokenizer = Tokenizer(basepipe.fields.groupby(ID))
    User, Item = tokenizer[USER], tokenizer[ITEM]
    model = LightGCN(
        tokenizer, basepipe.train().to_graph(User, Item)
    )
    model.load(basepipe.train().to_bigraph(User, Item))

    if cfg.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=cfg.lr, 
            momentum=cfg.momentum,
            nesterov=cfg.nesterov,
            weight_decay=cfg.weight_decay
        )
    elif cfg.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.lr,
            betas=(cfg.beta1, cfg.beta2),
            weight_decay=cfg.weight_decay
        )
    criterion = BPRLoss()

    coach = CoachForLightGCN(
        model=model,
        dataset=dataset,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=None,
        device=cfg.device
    )
    coach.compile(cfg, monitors=['loss', 'recall@10', 'recall@20', 'ndcg@10', 'ndcg@20'])
    coach.fit()



if __name__ == "__main__":
    main()

