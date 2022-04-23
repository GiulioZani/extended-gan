import torch as t
import ipdb


class Metrics:
    @staticmethod
    def mse(y, y_hat):
        diff_square = (y_hat.flatten() - y.flatten()) ** 2
        return IncrementalTuple(diff_square.sum(), diff_square.numel())

    @staticmethod
    def accuracy(y, y_hat):
        return IncrementalTuple((y == y_hat).sum(), y[0].numel())

    @staticmethod
    def precision(y_true, y_pred):
        TP = ((y_pred == 1) & (y_true == 1)).sum()
        FP = ((y_pred == 1) & (y_true == 0)).sum()
        return IncrementalTuple(TP, (TP + FP))

    @staticmethod
    def recall(y_true, y_pred):
        TP = ((y_pred == 1) & (y_true == 1)).sum()
        # FP = ((y_pred == 1) & (y_true == 0)).sum()
        FN = ((y_pred == 0) & (y_true == 1)).sum()
        return IncrementalTuple(TP, (TP + FN))


class IncrementalTuple:
    def __init__(self, val=None, denom=None):
        if val == None:
            self.val = t.tensor([0.0, 0.0])
        elif denom is not None:
            self.val = t.tensor((val, denom))
        else:
            self.val = val

    def reciprocal(self):
        return IncrementalTuple(t.tensor([self.val[1] - self.val[0], self.val[1]]))

    def __add__(self, x):
        return IncrementalTuple(x.val + self.val)

    def __iadd__(self, x):
        self.val += x.val
        return self

    def item(self):
        return (self.val[0] / self.val[1]).item()

    def __str__(self):
        return f"{self.item()}"

    def __format__(self, x):
        return self.item().__format__(x)


class MetricsManager:
    def __init__(
        self,
        metrics_names: tuple[str, ...],
        *,
        prefix: str = "",
        discretizing_threshold=0.5,
    ):
        self.discretizing_threshold = discretizing_threshold
        self.discrete_metrics = ("accuracy", "precision", "recall")
        self.prefix = prefix
        self.metrics = {}
        for name in metrics_names:
            self.metrics[name] = IncrementalTuple()

    def update(self, y, y_hat):
        discrete_y = y < self.discretizing_threshold
        discrete_y_hat = y_hat < self.discretizing_threshold

        for key, val in self.metrics.items():
            if key in self.discrete_metrics:
                val += Metrics.__dict__[key].__func__(discrete_y, discrete_y_hat)
            else:
                val += Metrics.__dict__[key].__func__(y, y_hat)

    def results(self):
        return {f"{self.prefix}_{key}": val.item() for key, val in self.metrics.items()}


def get_metrics(y, y_hat, mean):
    y = t.clone(y.cpu())
    y_hat = t.clone(y_hat.cpu())
    y[y < mean] = 0
    y[y >= mean] = 1
    y_hat[y_hat < mean] = 0
    y_hat[y_hat >= mean] = 1
    acc = accuracy(y, y_hat)
    prec = precision(y, y_hat)
    # if prec == t.nan:
    #    ipdb.set_trace()
    rec = recall(y, y_hat)
    return acc, prec, rec


def denormalize(x, mean, var):
    mean = t.mean(mean)
    var = t.var(var)
    return x * var + mean
