import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np

from src.model import UniversalApproximator
from src.create_gif import create_gif


def prepare_dataloader(
    func,
    x_range=(-5, 5),
    num_samples=1000,
    batch_size=32,
):
    x = torch.linspace(*x_range, num_samples).view(-1, 1)
    y = func(x)
    dataset = torch.utils.data.TensorDataset(x, y)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


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

    model = UniversalApproximator(
        hidden_dim=hidden_dim,
        lr=learning_rate,
        gif_pred=torch.linspace(*x_range, num_samples).view(-1, 1),
        model_name=func_name,
    )
    trainer = pl.Trainer(max_epochs=epochs, log_every_n_steps=10, enable_checkpointing=False)
    trainer.fit(model, train_loader)

    # Generate predictions
    x_test = torch.linspace(*x_test_range, num_samples).view(-1, 1)
    y_test = func(x_test)
    y_pred = model(x_test).detach()

    # Plot
    fig = plt.figure(figsize=(8, 5))
    plt.plot(x_test, y_test, label=f"True {func_name}", linewidth=2)
    plt.plot(x_test, y_pred, label=f"Predicted {func_name}", linestyle="--", linewidth=2)
    plt.legend()
    plt.title(f"Universal Approximation for {func_name} Function")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    
    return fig


if __name__ == "__main__":
    curves = {
        "sin": {"range_x": (-1.5 * np.pi, 1.5 * np.pi), "range_x_pred": (-2 * np.pi, 2 * np.pi), "func": torch.sin},
        "cos": {"range_x": (-1.5 * np.pi, 1.5 * np.pi), "range_x_pred": (-2 * np.pi, 2 * np.pi), "func": torch.cos},
        "tanh": {"range_x": (-1.5 * np.pi, 1.5 * np.pi), "range_x_pred": (-2 * np.pi, 2 * np.pi), "func": torch.tanh},
        "log": {"range_x": (0.001, 1.5), "range_x_pred": (-0.5, 2), "func": torch.log},
        "sqrt": {"range_x": (0, 1.5), "range_x_pred": (-0.5, 2), "func": torch.sqrt},
        "exp": {"range_x": (-1.5, 1.5), "range_x_pred": (-2, 2), "func": torch.exp},
        "x2": {"range_x": (-1.5, 1.5), "range_x_pred": (-2, 2), "func": lambda x: x**2},
        "abs": {"range_x": (-1.5, 1.5), "range_x_pred": (-2, 2), "func": torch.abs},
    }
    for curve_name, ranges in curves.items():
        range_x = ranges["range_x"]
        range_x_pred = ranges["range_x_pred"]
        func = ranges["func"]
        fig = train_and_visualize(
            func=func,
            func_name=curve_name,
            x_range=range_x,
            x_test_range=range_x_pred,
        )
        
        fig.savefig(f"src/plots/{curve_name}.png")
        create_gif(
            f"src/plots/{curve_name}",
            f"src/plots/{curve_name}.gif",
            duration=400,
        )
