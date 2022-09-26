

from typing import List, Dict

import torch

import freerec
from freerec.dict2obj import Config
from freerec.parser import Parser
from freerec.launcher import Coach
from freerec.models import NeuCF, RecSysArch
from freerec.criterions import BCELoss, BPRLoss
from freerec.data.datasets import MovieLens1M
from freerec.data.fields import Tokenizer
from freerec.data.tags import FEATURE, SPARSE, DENSE, USER, ITEM, ID, TARGET
from freerec.data.fields import SparseField, DenseField
from freerec.data.preprocessing import Binarizer

assert freerec.__version__ == "0.0.15", "Version Not Match, 0.0.15 required ..."

cfg = Parser()
cfg.add_argument("-eb", "--embedding_dim", type=int, default=8)
cfg.add_argument("-neg", "--num_negs", type=int, default=1)
cfg.add_argument("--criterion", type=str, default='BPRLoss')
cfg.add_argument("--regularizer", type=str, default='l2')
cfg.set_defaults(
    description="PureEmb",
    root="../../data/MovieLens1M",
    epochs=20,
    batch_size=256,
    optimizer='adam',
    lr=1e-3,
    weight_decay=1e-4
)
cfg.compile()




class MovieLens1M_(MovieLens1M):
    """
    MovieLens1M: (user, item, rating, timestamp)
        https://github.com/openbenchmark/BARS/tree/master/candidate_matching/datasets
    """

    _cfg = Config(
        sparse = [
            SparseField(name='UserID', na_value=0, dtype=int, tags=[USER, ID]),
            SparseField(name='ItemID', na_value=0, dtype=int, tags=[ITEM, ID]),
        ],
        dense = [DenseField(name="Timestamp", na_value=0., dtype=float, transformer='none', tags=FEATURE)],
        # target: 0|1
        target = [DenseField(name='Rating', na_value=None, dtype=int, transformer=Binarizer(threshold=1), tags=TARGET)]
    )

    _cfg.fields = _cfg.sparse + _cfg.target + _cfg.dense



class PurEmbd(RecSysArch):

    def __init__(self, tokenizer: Tokenizer) -> None:
        super().__init__()

        self.User = tokenizer.groupby(USER, ID)[0]
        self.Item = tokenizer.groupby(ITEM, ID)[0]

    def preprocess(self, users: Dict[str, torch.Tensor], items: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        users = users[self.User.name]
        items = items[self.Item.name]
        users = users.repeat((1, items.size(1)))
        return {self.User.name: users, self.Item.name: items}

    def forward(self, users: Dict[str, torch.Tensor], items: Dict[str, torch.Tensor]) -> torch.Tensor:
        inputs = self.preprocess(users, items)
        userEmbs = self.User.look_up(inputs[self.User.name])
        itemEmbs = self.Item.look_up(inputs[self.Item.name])
        outs = (userEmbs * itemEmbs).sum(dim=-1)
        if self.training:
            return outs
        else:
            return outs.sigmoid()

    def regularize(self, users: Dict[str, torch.Tensor], items: Dict[str, torch.Tensor], rtype: str = 'l2') -> Dict[str, torch.Tensor]:
        userEmbs = self.User.look_up(users[self.User.name]).view(-1, 1)
        itemEmbs = self.Item.look_up(items[self.Item.name]).view(-1, 1)
        if rtype == 'l2':
            loss = (userEmbs.pow(2).sum(-1).mean() + itemEmbs.pow(2).sum(-1).mean()) / 2
        elif rtype == 'l1':
            loss = (userEmbs.abs().sum(-1).mean() + itemEmbs.pow(2).sum(-1).mean()) / 2
        elif rtype == 'ortho':
            def ortho_regularizer(matrix: torch.Tensor):
                n = matrix.size(0)
                indices = torch.eye(n, n, dtype=torch.bool)
                return matrix[~indices].pow(2).mean() / 2
            userEmbs = self.User.embeddings.weight
            itemEmbs = self.Item.embeddings.weight
            loss = 0
            loss += ortho_regularizer(userEmbs.T @ userEmbs)
            loss += ortho_regularizer(itemEmbs.T @ itemEmbs)
        return loss



class CoachForNCF(Coach):


    def train_per_epoch(self):
        self.model.train()
        self.dataset.train()
        Target = self.fields.whichis(TARGET)
        for users, items, targets in self.dataloader:
            users = {name: val.to(self.device) for name, val in users.items()}
            items = {name: val.to(self.device) for name, val in items.items()}
            targets = targets[Target.name].to(self.device)

            preds = self.model(users, items)
            m, n = preds.size()
            targets = targets.repeat((1, n))
            targets[:, 1:].fill_(0)
            if cfg.criterion =='BPRLoss':
                loss = self.criterion(preds[:, 0], preds[:, 1])
            else:
                loss = self.criterion(preds, targets)
            loss += self.model.regularize(users, items, rtype=self.cfg.regularizer) * self.cfg.weight_decay

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.monitor(loss.item(), n=preds.size(0), mode="mean", prefix='train', pool=['LOSS'])


    def evaluate(self, prefix: str = 'valid'):
        self.model.eval()
        running_preds: List[torch.Tensor] = []
        running_targets: List[torch.Tensor] = []
        Target = self.fields.whichis(TARGET)
        for users, items, targets in self.dataloader:
            users = {name: val.to(self.device) for name, val in users.items()}
            items = {name: val.to(self.device) for name, val in items.items()}
            targets = targets[Target.name].to(self.device)

            preds = self.model(users, items)
            m, n = preds.size()
            targets = targets.repeat((1, n))
            targets[:, 1:].fill_(0)
            if cfg.criterion == 'BPRLoss':
                loss = self.criterion(preds[:, 0], preds[:, 1])
            else:
                loss = self.criterion(preds[:, :2], targets[:, :2])

            running_preds.append(preds.detach().clone().cpu())
            running_targets.append(targets.detach().clone().cpu())

            self.monitor(loss, n=m, mode="mean", prefix=prefix, pool=['LOSS'])

        running_preds = torch.cat(running_preds)
        running_targets = torch.cat(running_targets)
        self.monitor(
            running_preds, running_targets, 
            n=m, mode="mean", prefix=prefix,
            pool=['NDCG', 'PRECISION', 'RECALL', 'HITRATE', 'MSE', 'MAE', 'RMSE']
        )



def main():

    basepipe = MovieLens1M_(cfg.root).pin_(buffer_size=cfg.buffer_size).shard_()
    trainpipe = basepipe.negatives_for_train_(cfg.num_negs)
    validpipe = basepipe.negatives_for_eval_(99) # 1:99
    dataset = trainpipe.wrap_(validpipe).chunk_(batch_size=cfg.batch_size).dict_().tensor_().group_()

    tokenizer = Tokenizer(basepipe.fields)
    tokenizer.embed(
        cfg.embedding_dim, ID
    )
    model = PurEmbd(tokenizer).to(cfg.device)

    if cfg.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=cfg.lr, 
            momentum=cfg.momentum,
            nesterov=cfg.nesterov,
            # weight_decay=cfg.weight_decay
        )
    elif cfg.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.lr,
            betas=(cfg.beta1, cfg.beta2),
            # weight_decay=cfg.weight_decay
        )
    if cfg.criterion == 'BPRLoss':
        criterion = BPRLoss()
    else:
        criterion = BCELoss()

    coach = CoachForNCF(
        model=model,
        dataset=dataset,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=None,
        device=cfg.device
    )
    coach.compile(cfg, monitors=['loss', 'precision@10', 'recall@10', 'hitrate@10', 'ndcg@10', 'ndcg@20'])
    coach.fit()

if __name__ == "__main__":
    main()


