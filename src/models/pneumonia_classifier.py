from pytorch_lightning import LightningModule
from torch import nn, optim
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    BinaryStatScores,
)
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large


class PneumoniaClassifier(LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

        # 1. Setup model
        self.model = mobilenet_v3_large(weights="DEFAULT")

        # 1.1. Freeze all layers except last block
        for feature in self.model.features[:-1]:
            for param in feature.parameters():
                param.requires_grad = False

        # 1.3. Add final classification layer
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=960, out_features=1, bias=True),
            nn.Sigmoid(),
        )

        # 4. Set metrics
        self.loss = nn.BCELoss()
        self.accuracy = BinaryAccuracy()
        self.precision_score = BinaryPrecision()
        self.recall_score = BinaryRecall()
        self.stats_scores = BinaryStatScores()  # TP, FP, TN, FN
        self.f1_score = BinaryF1Score()
        self.auroc = BinaryAUROC()
        self.avg_precision = BinaryAveragePrecision()

    def get_required_transforms():
        return MobileNet_V3_Large_Weights.DEFAULT.transforms

    def training_step(self, batch, batch_idx):
        metrics = {
            f"train_{name}": metric
            for name, metric in self._shared_eval_step(batch, batch_idx).items()
        }
        self.log_dict(metrics)
        return metrics["train_loss"]

    def validation_step(self, batch, batch_idx):
        metrics = {
            f"val_{name}": metric
            for name, metric in self._shared_eval_step(batch, batch_idx).items()
        }
        self.log_dict(metrics)
        return metrics

    def test_step(self, batch, batch_idx):
        metrics = {
            f"test_{name}": metric
            for name, metric in self._shared_eval_step(batch, batch_idx).items()
        }
        self.log_dict(metrics)
        return metrics

    def _shared_eval_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x).squeeze(-1)  # remove the last dimension to match shapes

        stats_scores = self.stats_scores(y_hat, y).float()

        return {
            "loss": self.loss(y_hat, y),
            "accuracy": self.accuracy(y_hat, y.int()),
            "precision": self.precision_score(y_hat, y.int()),
            "recall": self.recall_score(y_hat, y.int()),
            "f1_score": self.f1_score(y_hat, y.int()),
            "auroc": self.auroc(y_hat, y.int()),
            "avg_precision": self.avg_precision(y_hat, y.int()),
            "true_positives": stats_scores[0],
            "false_positives": stats_scores[1],
            "true_negatives": stats_scores[2],
            "false_negatives": stats_scores[3],
        }

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self.model(batch[0])

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=0.02)
