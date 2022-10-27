

from typing import Dict, Optional

import torch
import torch_geometric.transforms as T
from torch_geometric.data.data import Data
from torch_geometric.nn import LGConv

from freerec.parser import Parser
from freerec.launcher import Coach
from freerec.models import LightGCN, RecSysArch
from freerec.criterions import BPRLoss
from freerec.data.datasets import Gowalla_m1
from freerec.data.fields import Tokenizer
from freerec.data.tags import USER, ITEM, ID


cfg = Parser()
cfg.add_argument("-eb", "--embedding-dim", type=int, default=64)
cfg.add_argument("--layers", type=int, default=3)
cfg.add_argument("--self-loops", type=str, default='True')
cfg.add_argument("--reg", type=str, default='True')
cfg.set_defaults(
    description="LightGCN",
    root="../../data/Gowalla",
    epochs=1000,
    batch_size=2048,
    optimizer='adam',
    lr=1e-3,
    weight_decay=1e-4,
    seed=2020
)
cfg.compile()


__all__ = ['LightGCN']


class LightGCN(RecSysArch):

    def __init__(
        self, tokenizer: Tokenizer, 
        graph: Data,
        num_layers: int = 3
    ) -> None:
        super().__init__()

        self.tokenizer = tokenizer
        self.conv = LGConv()
        self.num_layers = num_layers
        self.graph = graph

        self.User, self.Item = self.tokenizer[USER, ID], self.tokenizer[ITEM, ID]

        self.initialize()

    @property
    def graph(self):
        return self.__graph

    @graph.setter
    def graph(self, graph: Data):
        self.__graph = graph
        if cfg.self_loops:
            T.AddSelfLoops()(self.__graph)
        T.ToSparseTensor()(self.__graph)

    def forward(
        self, users: Optional[Dict[str, torch.Tensor]] = None, 
        items: Optional[Dict[str, torch.Tensor]] = None
    ):
        adj_t = self.graph.adj_t.to(self.device)
        userEmbs = self.User.embeddings.weight
        itemEmbs = self.Item.embeddings.weight
        features = torch.cat((userEmbs, itemEmbs), dim=0).flatten(1) # N x D
        avgFeats = features / (self.num_layers + 1)
        for _ in range(self.num_layers):
            features = self.conv(features, adj_t)
            avgFeats += features / (self.num_layers + 1)
        userFeats, itemFeats = torch.split(avgFeats, (self.User.count, self.Item.count))

        if self.training: # Batch
            users, items = self.broadcast(
                users[self.User.name], items[self.Item.name]
            )
            userFeats = userFeats[users] # B x n x D
            itemFeats = itemFeats[items] # B x n x D
            userEmbs = self.User.look_up(users) # B x n x D
            itemEmbs = self.Item.look_up(items) # B x n x D
            return torch.mul(userFeats, itemFeats).sum(-1), userEmbs, itemEmbs
        else:
            return userFeats, itemFeats



class CoachForLightGCN(Coach):


    def reg_loss(self, userEmbds, itemEmbds):
        userEmbds, itemEmbds = userEmbds.flatten(1), itemEmbds.flatten(1)
        loss = userEmbds.pow(2).sum() + itemEmbds.pow(2).sum() * 2
        loss = loss / userEmbds.size(0)
        return loss / 2

    def train_per_epoch(self):
        for users, items in self.dataloader:
            users = {name: val.to(self.device) for name, val in users.items()}
            items = {name: val.to(self.device) for name, val in items.items()}

            preds, users, items = self.model(users, items)
            pos, neg = preds[:, 0], preds[:, 1]
            if cfg.reg == 'True':
                reg_loss = self.reg_loss(users.flatten(1), items.flatten(1)) * self.cfg.weight_decay
            else:
                reg_loss = 0
            loss = self.criterion(pos, neg) + reg_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.monitor(loss.item(), n=preds.size(0), mode="mean", prefix='train', pool=['LOSS'])

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

    basepipe = Gowalla_m1(cfg.root)
    trainpipe = basepipe.uniform_sampling_(num_negatives=1).tensor_().chunk_(cfg.batch_size)
    validpipe = basepipe.trisample_(batch_size=cfg.batch_size).tensor_()
    dataset = trainpipe.wrap_(validpipe).group_((USER, ITEM))

    tokenizer = Tokenizer(basepipe.fields)
    tokenizer.embed(
        cfg.embedding_dim, ID
    )
    User, Item = tokenizer[USER], tokenizer[ITEM]
    model = LightGCN(
        tokenizer, basepipe.train().to_graph(User, Item)
    )

    weight_decay = 0 if cfg.reg == 'True' else cfg.weight_decay
    if cfg.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=cfg.lr, 
            momentum=cfg.momentum,
            nesterov=cfg.nesterov,
            weight_decay=weight_decay
        )
    elif cfg.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.lr,
            betas=(cfg.beta1, cfg.beta2),
            weight_decay=weight_decay
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

