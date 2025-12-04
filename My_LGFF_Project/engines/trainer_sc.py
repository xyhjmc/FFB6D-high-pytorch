class TrainerSC:
    def __init__(self, model, optimizer, loss_fn, train_loader, val_loader, cfg):
        ...

    def train(self):
        for epoch in range(cfg.num_epochs):
            self.train_one_epoch(epoch)
            if epoch % cfg.val_interval == 0:
                self.validate(epoch)
            # TODO: 保存 checkpoint, log
