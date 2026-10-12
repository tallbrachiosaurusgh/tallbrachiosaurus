import matplotlib.pyplot as plt
import torch
import torch.nn as nn


class LinearRegression(nn.Module):
    def __init__(self, learning_rate=0.1, epochs=2000):
        super().__init__()

        self.learning_rate = learning_rate
        self.epochs = epochs

        # Parameters of the linear model, started at zero.
        # nn.Parameter registers them so the optimizer can find them.
        self.w0 = nn.Parameter(torch.zeros(1))
        self.w1 = nn.Parameter(torch.zeros(1))

        # SGD optimizer and mean-squared-error loss
        self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        self.loss_function = nn.MSELoss()

        # Training history, appended to once per epoch
        self.w0_history = []
        self.w1_history = []
        self.loss_history = []

        # Kept for analysis_plot so it can redraw the training data
        self.x_train = None
        self.y_train = None

    @staticmethod
    def _to_column(values):
        if not torch.is_tensor(values):
            values = torch.tensor(
                getattr(values, "values", values), dtype=torch.float32
            )
        return values.to(torch.float32).view(-1, 1)

    def forward(self, x):
        return self.w0 + self.w1 * x

    def fit(self, train, test):
        x_train = self._to_column(train[0])
        y_train = self._to_column(train[1])
        x_test = self._to_column(test[0])
        y_test = self._to_column(test[1])

        self.x_train, self.y_train = x_train, y_train

        for _ in range(self.epochs):
            # Clear gradients from the previous step; PyTorch accumulates
            # them by default, so skipping this would sum them up.
            self.optimizer.zero_grad()

            y_predicted = self.forward(x_train)
            loss = self.loss_function(y_predicted, y_train)

            loss.backward()     # backpropagate to get dLoss/dw0, dLoss/dw1
            self.optimizer.step()   # w <- w - learning_rate * gradient

            self.w0_history.append(self.w0.item())
            self.w1_history.append(self.w1.item())
            self.loss_history.append(loss.item())

        # Evaluate on the held-out test data. no_grad() because this is
        # scoring, not training, so no gradients are needed.
        with torch.no_grad():
            y_hat = self.forward(x_test)
            ss_residual = ((y_test - y_hat) ** 2).sum()
            ss_total = ((y_test - y_test.mean()) ** 2).sum()
            r2 = (1 - ss_residual / ss_total).item()

        print(f"Trained for {self.epochs} epochs "
              f"at learning rate {self.learning_rate}")
        print(f"  w0 (intercept) = {self.w0.item():.4f}")
        print(f"  w1 (slope)     = {self.w1.item():.4f}")
        print(f"  final training loss = {self.loss_history[-1]:.4f}")
        print(f"  R^2 on test data    = {r2:.4f}")

        return r2

    def predict(self, x):
        single = not hasattr(x, "__len__")
        x_tensor = self._to_column([x] if single else x)

        with torch.no_grad():
            predictions = self.forward(x_tensor).view(-1)

        return predictions.item() if single else predictions

    def analysis_plot(self):
        if not self.loss_history:
            raise RuntimeError("Call fit() before analysis_plot().")

        fig, ax = plt.subplots(2, 2, figsize=(13, 9))

        x = self.x_train.view(-1).numpy()
        y = self.y_train.view(-1).numpy()

        # Original data with the fitted regression line
        ax[0, 0].scatter(x, y, s=18, alpha=0.6, color='#2F9C95',
                         label='Training data')
        x_line = torch.linspace(float(x.min()), float(x.max()), 100)
        with torch.no_grad():
            y_line = self.forward(x_line.view(-1, 1)).view(-1).numpy()
        ax[0, 0].plot(x_line.numpy(), y_line, color='crimson', lw=2,
                      label='Fitted line')
        ax[0, 0].set_xlabel('BCR', fontsize=12)
        ax[0, 0].set_ylabel('Annual Production (MWh)', fontsize=12)
        ax[0, 0].set_title('Data and fitted model', fontsize=13)
        ax[0, 0].legend()

        # Loss curve, log scale because the first epochs dwarf the rest
        ax[0, 1].plot(self.loss_history, color='crimson', lw=1.5)
        ax[0, 1].set_yscale('log')
        ax[0, 1].set_xlabel('Epoch', fontsize=12)
        ax[0, 1].set_ylabel('MSE loss (log scale)', fontsize=12)
        ax[0, 1].set_title('Loss during training', fontsize=13)

        # Parameter trajectories
        ax[1, 0].plot(self.w0_history, color='#2F9C95', lw=1.5)
        ax[1, 0].set_xlabel('Epoch', fontsize=12)
        ax[1, 0].set_ylabel('$w_0$', fontsize=12)
        ax[1, 0].set_title(f'Intercept converging to {self.w0.item():.2f}',
                           fontsize=13)

        ax[1, 1].plot(self.w1_history, color='#E4A010', lw=1.5)
        ax[1, 1].set_xlabel('Epoch', fontsize=12)
        ax[1, 1].set_ylabel('$w_1$', fontsize=12)
        ax[1, 1].set_title(f'Slope converging to {self.w1.item():.2f}',
                           fontsize=13)

        for row in ax:
            for a in row:
                a.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()
        return fig


if __name__ == "__main__":
    import csv
    import urllib.request
 
    url = ("https://raw.githubusercontent.com/rahulbhadani/"
           "Datasets/master/Hydropower.csv")
    with urllib.request.urlopen(url) as response:
        rows = list(csv.DictReader(response.read().decode().splitlines()))
 
    x_all = [float(r['BCR']) for r in rows]
    y_all = [float(r['AnnualProduction']) for r in rows]
 
    split = int(0.75 * len(x_all))
    train = (x_all[:split], y_all[:split])
    test = (x_all[split:], y_all[split:])
 
    model = LinearRegression(learning_rate=0.1, epochs=2000)
    model.fit(train, test)
    print("\nPredicted AnnualProduction at BCR = 0.5:",
          round(model.predict(0.5), 2))
 
