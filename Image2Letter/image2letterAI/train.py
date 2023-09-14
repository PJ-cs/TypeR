from typing import Any
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT, TRAIN_DATALOADERS
from models import NeuralNetwork
from utils import TypeRLoss
from torch import optim


class LitModel(pl.LightningModule):
    def __init__(self, 
                font_path : str,
                transposed_kernel_size : int,
                transposed_stride : int,
                max_letter_per_pix: int,
                letters : list[str],
                lr : float,
                alpha: float,
                beta: float,
                gamma: float                
                ) -> None:
        super().__init__()
        self.lr = lr
        self.model = NeuralNetwork(font_path, transposed_kernel_size, transposed_stride, max_letter_per_pix, letters, 1/255.)
        self.loss = TypeRLoss(max_letter_per_pix, alpha, beta, gamma)

    def training_step(self, batch, batch_idx : int) -> STEP_OUTPUT:
        img_in, img_target = batch
        key_strokes, out_img = self.model(img_in)
        loss = self.loss.forward(key_strokes, out_img, img_target)
        self.log("train_loss", float(loss))
        return loss
    
    def configure_optimizers(self) -> Any:
        optimizer = optim.Adam(self.parameters, lr=self.lr)
        return optimizer
