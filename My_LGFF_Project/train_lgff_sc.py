# lgff/train_lgff_sc.py
"""
最后让 Codex 写两个很薄的脚本，负责：

解析命令行和 config；

构建 dataset, dataloader；

构建 model, optimizer, scheduler；

构建 loss, trainer；

调 trainer.train()。
"""
from lgff.models.lgff_sc import LGFF_SC
from lgff.datasets.single_loader import SingleObjectDataset
from lgff.losses import LGFFLoss
from lgff.engines.trainer_sc import TrainerSC
from common.config import load_config  # TODO: 从 common 里引

def main():
    cfg = load_config()
    # TODO: 构建 dataset / dataloader
    # TODO: 构建模型
    # TODO: 构建优化器、loss、trainer
    # trainer.train()

if __name__ == "__main__":
    main()
