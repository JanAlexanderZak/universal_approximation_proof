from pathlib import Path

import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt


PLOTS_DIR = Path(__file__).parent / "plots"

Y_LIMITS = {
    "sin": (-2, 2),
    "cos": (-2, 2),
    "tanh": (-2, 2),
    "abs": (0, 2),
    "x2": (0, 2),
    "exp": (0, 4),
    "log": (-10, 2),
    "sqrt": (-0.5, 1.5),
    "heaviside": (-0.5, 1.5),
    "sawtooth": (-1.5, 1.5),
    "weierstrass": (-2, 2),
    "sin_cos_2d": (-1.5, 1.5),
}


class UniversalApproximator(pl.LightningModule):
    def __init__(
        self,
        input_dim=1,
        hidden_dim=64,
        output_dim=1,
        lr=1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class VisualizationCallback2D(pl.Callback):
    def __init__(self, x1_grid, x2_grid, func, func_name, every_n_epochs=50):
        self.x1_grid = x1_grid
        self.x2_grid = x2_grid
        x_flat = torch.stack(
            [x1_grid.flatten(), x2_grid.flatten()], dim=1
        )
        self.x_flat = x_flat
        self.y_true = func(x_flat).view(x1_grid.shape)
        self.func_name = func_name
        self.every_n_epochs = every_n_epochs
        self.output_dir = PLOTS_DIR / func_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs != 0:
            return

        y_pred = pl_module(self.x_flat).detach().view(self.x1_grid.shape)

        fig = plt.figure(figsize=(14, 5))

        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax1.plot_surface(
            self.x1_grid.numpy(), self.x2_grid.numpy(), self.y_true.numpy(),
            cmap="viridis", alpha=0.7,
        )
        ax1.set_title(f"True {self.func_name}")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_zlabel("z")

        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        ax2.plot_surface(
            self.x1_grid.numpy(), self.x2_grid.numpy(), y_pred.numpy(),
            cmap="plasma", alpha=0.7,
        )
        ax2.set_title(
            f"Predicted {self.func_name} "
            f"(epoch {trainer.current_epoch})"
        )
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_zlabel("z")

        fig.savefig(self.output_dir / f"{trainer.current_epoch}.png", dpi=100)
        plt.close(fig)


class VisualizationCallback(pl.Callback):
    def __init__(self, x_plot, func, func_name, every_n_epochs=50):
        self.x_plot = x_plot
        self.y_true = func(x_plot)
        self.func_name = func_name
        self.every_n_epochs = every_n_epochs
        self.output_dir = PLOTS_DIR / func_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs != 0:
            return

        y_pred = pl_module(self.x_plot).detach()

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            self.x_plot, self.y_true,
            label=f"True {self.func_name}", linewidth=2,
        )
        ax.plot(
            self.x_plot, y_pred,
            label=f"Predicted {self.func_name}",
            linestyle="--", linewidth=2,
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend()
        ax.grid()

        if self.func_name in Y_LIMITS:
            ax.set_ylim(*Y_LIMITS[self.func_name])

        fig.savefig(self.output_dir / f"{trainer.current_epoch}.png")
        plt.close(fig)
