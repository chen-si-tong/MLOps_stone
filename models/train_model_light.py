import torch
import torch.nn.functional as F
from torch import optim
from model_light import MNISTModel
import pickle
import matplotlib.pyplot as plt
import hydra
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


def train(config):
    with open(config.train_data_path, "rb") as file:
        train_set = pickle.load(file)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    checkpoint_callback = ModelCheckpoint(
            monitor='train_loss',  # 监控的指标，可以根据需要更改
            dirpath="/Users/chensitong/MLOps/project/MLOps_test/checkpoints",  # 指定保存模型的目录
            filename='best_model',  # 模型文件名
            save_top_k=1,  # 保存最佳模型的数量
            mode='min',  # 以最小值为目标，也可以是'max'等
    )
    model = MNISTModel(config)  # 传递适当的配置参数
    trainer = pl.Trainer(
        max_epochs=3,  
        default_root_dir="/Users/chensitong/MLOps/project/MLOps_test/checkpoints", # 指定checkpoints文件保存的目录
        limit_train_batches=0.8, #使用部分数据进行训练
        logger=pl.loggers.WandbLogger(project="mnist",name="test_1"),
    )
    
    # 使用 Trainer 进行训练
    trainer.fit(model, train_loader)
    torch.save(model.state_dict(), "final_model/mnist_model.pth")


def evaluate(config):

    with open(config.test_data_path, "rb") as file:
        test_set = pickle.load(file)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=config.batch_size)

    # 加载已保存的最佳模型
    
    model = MNISTModel(config)
    model.load_state_dict(torch.load(config.model_checkpoint))

    # 设置模型为评估模式
    model.eval()
    trainer = pl.Trainer()
    trainer.test(model,test_loader)


@hydra.main(config_path="/Users/chensitong/MLOps/project/MLOps_test/conf", config_name="light.yaml",version_base="1.3.2")
def main(config): #这里控制train还是evaluate
    print(f"configuration: \n {OmegaConf.to_yaml(config)}")
    train(config)
    evaluate(config)


if __name__ == "__main__":
    main()
    


