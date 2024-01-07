from torch import nn,optim
import torch.nn.functional as F
from pytorch_lightning import LightningModule
import wandb

class MNISTModel(LightningModule):
    def __init__(self,cfg):
        super(MNISTModel, self).__init__()
        self.cfg = cfg 
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=cfg.kernel_size, padding=cfg.padding),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=cfg.kernel_size, padding=cfg.padding),
            nn.ReLU(),
            nn.MaxPool2d(2)      
        )


        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.out_features)
        )

        self.criterium = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.classifier(self.backbone(x))
    
    def training_step(self,batch,batch_idx):
        data,target = batch
        preds = self(data)
        loss = self.criterium(preds,target)
        acc = (target == preds.argmax(dim=-1)).float().mean()
        self.log('train_loss', loss)
        self.log('train_acc', acc)

        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(),lr=self.cfg.lr)
    
    def test_step(self,batch,batch_idx):
        data,target = batch
        preds = self(data)
        loss = self.criterium(preds,target)
        acc = (target == preds.argmax(dim=-1)).float().mean()
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        return loss