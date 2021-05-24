import torch
import pytorch_lightning as pl

class nn_model(pl.LightningModule):

    def __init__(self, n_in, n_out, optimizer, lr, objective, validation_loss):
        super(nn_model, self).__init__()
        self.optimizer = optimizer
        self.lr = lr
        self.n_in = n_in
        self.n_out = n_out

        self.linear = torch.nn.Linear(n_in, n_out)
        self.objective = objective()
        self.validation_loss = validation_loss()


    def forward(self, x):
        x = self.linear(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        train_loss = self.objective(y_pred, y)
        self.log('train_loss', train_loss)
        return (train_loss)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        val_loss = self.validation_loss(y_pred, y)
        self.log('val_loss', val_loss.item())
        return (val_loss)

    def configure_optimizers(self):
        if self.optimizer == 'AdamW':
            return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer == 'Adadelta':
            return torch.optim.Adadelta(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer == 'Adagrad':
            return torch.optim.Adagrad(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer == 'Adam':
            return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer == 'Adamax':
            return torch.optim.Adamax(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer == 'ASGD':
            return torch.optim.ASGD(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer == 'LBFGS':
            return torch.optim.LBFGS(self.parameters(), lr=self.lr)
        elif self.optimizer == 'RMSprop':
            return torch.optim.RMSprop(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer == 'Rprop':
            return torch.optim.Rprop(self.parameters(), lr=self.lr)
        elif self.optimizer == 'SGD':
            return torch.optim.AdamW(self.parameters(), lr=self.lr)