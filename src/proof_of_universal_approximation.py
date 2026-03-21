import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np

from src.model import (
    PLOTS_DIR,
    UniversalApproximator,
    VisualizationCallback,
    VisualizationCallback2D,
)
from src.create_gif import create_gif
from src.target_functions import heaviside, sawtooth, weierstrass, sin_cos_2d


def prepare_dataloader(
    func,
    x_range=(-5, 5),
    num_samples=1000,
    batch_size=32,
):
    x = torch.linspace(*x_range, num_samples).view(-1, 1)
    y = func(x)
    dataset = torch.utils.data.TensorDataset(x, y)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
    )


def train_and_visualize(
    func,
    func_name,
    x_range=(-1, 1),
    x_test_range=(-2, 2),
    num_samples=300,
    hidden_dim=64,
    learning_rate=1e-4,
    epochs=2000,
):
    train_loader = prepare_dataloader(func, x_range, num_samples)

    x_plot = torch.linspace(*x_range, num_samples).view(-1, 1)
    viz_callback = VisualizationCallback(
        x_plot=x_plot,
        func=func,
        func_name=func_name,
    )

    model = UniversalApproximator(hidden_dim=hidden_dim, lr=learning_rate)
    trainer = pl.Trainer(
        max_epochs=epochs,
        log_every_n_steps=10,
        enable_checkpointing=False,
        callbacks=[viz_callback],
    )
    trainer.fit(model, train_loader)

    # Generate predictions
    x_test = torch.linspace(*x_test_range, num_samples).view(-1, 1)
    y_test = func(x_test)
    y_pred = model(x_test).detach()

    # Compute test MSE
    test_mse = torch.nn.functional.mse_loss(y_pred, y_test).item()
    print(f"{func_name}: test MSE = {test_mse:.6f}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        x_test, y_test,
        label=f"True {func_name}", linewidth=2,
    )
    ax.plot(
        x_test, y_pred,
        label=f"Predicted {func_name}",
        linestyle="--", linewidth=2,
    )
    ax.legend()
    ax.set_title(f"Universal Approximation for {func_name} Function")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid()

    return fig


def prepare_dataloader_2d(
    func, x_range, y_range, num_samples=50, batch_size=64,
):
    x1 = torch.linspace(*x_range, num_samples)
    x2 = torch.linspace(*y_range, num_samples)
    grid_x1, grid_x2 = torch.meshgrid(
        x1, x2, indexing="ij",
    )
    x_data = torch.stack(
        [grid_x1.flatten(), grid_x2.flatten()], dim=1,
    )
    y = func(x_data)
    dataset = torch.utils.data.TensorDataset(x_data, y)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
    )


def train_and_visualize_2d(
    func,
    func_name,
    x_range=(-np.pi, np.pi),
    y_range=(-np.pi, np.pi),
    num_samples=50,
    hidden_dim=128,
    learning_rate=1e-3,
    epochs=3000,
):
    train_loader = prepare_dataloader_2d(func, x_range, y_range, num_samples)

    x1 = torch.linspace(*x_range, num_samples)
    x2 = torch.linspace(*y_range, num_samples)
    x1_grid, x2_grid = torch.meshgrid(x1, x2, indexing="ij")

    viz_callback = VisualizationCallback2D(
        x1_grid=x1_grid,
        x2_grid=x2_grid,
        func=func,
        func_name=func_name,
    )

    model = UniversalApproximator(
        input_dim=2, hidden_dim=hidden_dim, lr=learning_rate,
    )
    trainer = pl.Trainer(
        max_epochs=epochs,
        log_every_n_steps=10,
        enable_checkpointing=False,
        callbacks=[viz_callback],
    )
    trainer.fit(model, train_loader)

    # Final evaluation
    x_flat = torch.stack(
        [x1_grid.flatten(), x2_grid.flatten()], dim=1,
    )
    y_pred = model(x_flat).detach().view(x1_grid.shape)
    y_true = func(x_flat).view(x1_grid.shape)

    test_mse = torch.nn.functional.mse_loss(y_pred, y_true).item()
    print(f"{func_name}: test MSE = {test_mse:.6f}")

    fig = plt.figure(figsize=(14, 5))

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(
        x1_grid.numpy(), x2_grid.numpy(),
        y_true.numpy(), cmap="viridis", alpha=0.7,
    )
    ax1.set_title(f"True {func_name}")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.plot_surface(
        x1_grid.numpy(), x2_grid.numpy(),
        y_pred.numpy(), cmap="plasma", alpha=0.7,
    )
    ax2.set_title(f"Predicted {func_name}")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("z")

    return fig


def main():
    pi = np.pi
    curves = {
        "sin": {
            "range_x": (-1.5 * pi, 1.5 * pi),
            "range_x_pred": (-2 * pi, 2 * pi),
            "func": torch.sin,
        },
        "cos": {
            "range_x": (-1.5 * pi, 1.5 * pi),
            "range_x_pred": (-2 * pi, 2 * pi),
            "func": torch.cos,
        },
        "tanh": {
            "range_x": (-3, 3),
            "range_x_pred": (-4, 4),
            "func": torch.tanh,
        },
        "log": {
            "range_x": (0.001, 1.5),
            "range_x_pred": (0.001, 2),
            "func": torch.log,
        },
        "sqrt": {
            "range_x": (0.001, 1.5),
            "range_x_pred": (0.001, 2),
            "func": torch.sqrt,
        },
        "exp": {
            "range_x": (-1.5, 1.5),
            "range_x_pred": (-2, 2),
            "func": torch.exp,
        },
        "x2": {
            "range_x": (-1.5, 1.5),
            "range_x_pred": (-2, 2),
            "func": lambda x: x**2,
        },
        "abs": {
            "range_x": (-1.5, 1.5),
            "range_x_pred": (-2, 2),
            "func": torch.abs,
        },
        # Pathological functions
        "heaviside": {
            "range_x": (-2, 2),
            "range_x_pred": (-3, 3),
            "func": heaviside,
            "hidden_dim": 128,
            "epochs": 4000,
            "lr": 1e-3,
            "num_samples": 1000,
        },
        "sawtooth": {
            "range_x": (-2 * pi, 2 * pi),
            "range_x_pred": (-3 * pi, 3 * pi),
            "func": sawtooth,
            "hidden_dim": 128,
            "epochs": 4000,
            "lr": 1e-3,
            "num_samples": 1000,
        },
        "weierstrass": {
            "range_x": (-2, 2),
            "range_x_pred": (-2.5, 2.5),
            "func": weierstrass,
            "hidden_dim": 256,
            "epochs": 5000,
            "lr": 5e-4,
            "num_samples": 1000,
        },
    }
    for curve_name, config in curves.items():
        fig = train_and_visualize(
            func=config["func"],
            func_name=curve_name,
            x_range=config["range_x"],
            x_test_range=config["range_x_pred"],
            hidden_dim=config.get("hidden_dim", 64),
            learning_rate=config.get("lr", 1e-4),
            epochs=config.get("epochs", 2000),
            num_samples=config.get("num_samples", 300),
        )

        fig.savefig(PLOTS_DIR / f"{curve_name}.png")
        plt.close(fig)
        create_gif(
            PLOTS_DIR / curve_name,
            PLOTS_DIR / f"{curve_name}.gif",
            duration=400,
        )

    # 2D demonstrations
    surfaces = {
        "sin_cos_2d": {
            "func": sin_cos_2d,
            "x_range": (-np.pi, np.pi),
            "y_range": (-np.pi, np.pi),
            "hidden_dim": 128,
            "epochs": 3000,
            "lr": 1e-3,
        },
    }
    for name, config in surfaces.items():
        fig = train_and_visualize_2d(
            func=config["func"],
            func_name=name,
            x_range=config["x_range"],
            y_range=config["y_range"],
            hidden_dim=config.get("hidden_dim", 128),
            learning_rate=config.get("lr", 1e-3),
            epochs=config.get("epochs", 3000),
        )
        fig.savefig(PLOTS_DIR / f"{name}.png", dpi=100)
        plt.close(fig)
        create_gif(
            PLOTS_DIR / name,
            PLOTS_DIR / f"{name}.gif",
            duration=400,
        )


if __name__ == "__main__":
    main()
