import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt


class UniversalApproximator(pl.LightningModule):
    def __init__(
        self,
        gif_pred,
        model_name,
        input_dim=1,
        hidden_dim=64,
        output_dim=1,
        lr=1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.gif_pred = gif_pred
        self.model_name = model_name
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim)
        )
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        self.log("train_loss", loss)
        
        if self.trainer.current_epoch % 50 == 0:
            fig = plt.figure(figsize=(8, 5))
            y_pred = self(self.gif_pred).detach()
            plt.plot(self.gif_pred, y_pred, linestyle="--", linewidth=2)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.grid()
            if (self.model_name == "sin") or (self.model_name == "cos") or (self.model_name == "tanh"):
                plt.ylim(-2, 2)
            elif (self.model_name == "abs") or (self.model_name == "x2"):
                plt.ylim(0, 2)
            elif self.model_name == "exp":
                plt.ylim(0, 4)
            elif self.model_name == "log":
                plt.ylim(-10, 2)
            elif self.model_name == "sqrt":
                plt.ylim(-0.5, 1.5)

            fig.savefig(f"src/plots/{self.model_name}/{self.trainer.current_epoch}.png")
            plt.clf()
            plt.close()
        
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)