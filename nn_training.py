import torch
import pytorch_lightning as pl

class nn_training:
    def __init__(self, nn_model, X, y, device = torch.device('cuda:0')):

        self.X = X
        self.y = y
        self.nn_model = nn_model
        self.device = device

    def data_loaders(self, X, y):
        X_train = torch.from_numpy(X).float().to(self.device)
        y_train = torch.from_numpy(y).float().to(self.device)
        X_test = torch.from_numpy(X).float().to(self.device)
        y_test = torch.from_numpy(y).float().to(self.device)

        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=X_train.shape[0],
                                                   shuffle=False)

        val_dataset = torch.utils.data.TensorDataset(X_test, y_test)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=y_test.shape[0],
                                                 shuffle=False)

        return (train_loader, val_loader)

    def train(self, params_trans, val_fold=True):

        class MetricsCallback(pl.Callback):
            """PyTorch Lightning metric callback."""

            def __init__(self):
                super().__init__()
                self.metrics = []

            def on_validation_end(self, trainer, pl_module):
                self.metrics.append(trainer.callback_metrics)

        train_loader, val_loader = self.data_loaders(self.X, self.y)
        metrics_callback = MetricsCallback()
        if val_fold == True:
            trainer = pl.Trainer(min_epochs=params_trans['min_epochs'],
                                 max_epochs=params_trans['max_epochs'],
                                 progress_bar_refresh_rate=0,
                                 callbacks=[pl.callbacks.early_stopping.EarlyStopping(min_delta=0.00000000001,
                                                                                      patience=3,
                                                                                      monitor='val_loss'),
                                            metrics_callback],
                                 num_sanity_val_steps=0,
                                 gpus=1)
        else:
            trainer = pl.Trainer(min_epochs=params_trans['min_epochs'],
                                 max_epochs=params_trans['max_epochs'],
                                 progress_bar_refresh_rate=0,
                                 num_sanity_val_steps=0,
                                 gpus=1)
        params_trans.pop('min_epochs', None)
        params_trans.pop('max_epochs', None)
        my_model = self.nn_model(**params_trans)
        trainer.fit(my_model, train_loader, val_loader)
        print(metrics_callback.metrics)

        return (my_model.val_losses)