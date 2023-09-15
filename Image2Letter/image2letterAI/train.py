from typing import Any, Optional
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT
from models import NeuralNetwork
from utils import TypeRLoss
from torch import optim
import torchvision
import config.config as config

class LitModel(pl.LightningModule):
    def __init__(self, 
                font_path : str,
                transposed_kernel_size : int,
                transposed_stride : int,
                transposed_padding : int,
                max_letter_per_pix: int,
                letters : list[str],
                lr : float,
                alpha: float,
                beta: float,
                gamma: float                
                ) -> None:
        super().__init__()
        self.lr = lr
        self.model = NeuralNetwork(font_path, transposed_kernel_size, transposed_stride, transposed_padding, max_letter_per_pix, letters, 1/255.)
        self.loss = TypeRLoss(max_letter_per_pix, alpha, beta, gamma)

    def training_step(self, batch, batch_idx : int) -> STEP_OUTPUT:
        img_in, img_target, label = batch
        out_img, key_strokes = self.model(img_in)
        loss = self.loss.forward(key_strokes, out_img, img_target)
        self.log("train_loss", float(loss))
        return loss
    
    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT | None:
        img_in, img_target, label = batch
        out_img, key_strokes = self.model(img_in)
        loss = self.loss.forward(key_strokes, out_img, img_target)
        self.log("val_loss", float(loss))
        if batch_idx % 6 == 0:
            grid = torchvision.utils.make_grid(out_img[:4])
            self.logger.experiment.add_image('generated_images', grid, 0)

        return loss
    
    def test_step(self, batch, batch_idx) -> STEP_OUTPUT | None:
        img_in, img_target, label = batch
        out_img, key_strokes = self.model(img_in)
        loss = self.loss.forward(key_strokes, out_img, img_target)
        self.log("test_loss", float(loss))
        if batch_idx % 6 == 0:
            grid = torchvision.utils.make_grid(out_img[:4])
            self.logger.experiment.add_image('generated_images', grid, 0)
        return loss
    
    def configure_optimizers(self) -> Any:
        optimizer = optim.Adam(self.parameters, lr=self.lr)
        return optimizer

net = LitModel(str(config.FONT_PATH), 64, round(64*0.035), 31, 5, config.TYPEWRITER_CONFIG["letterList"], lr=0.0001, alpha=1.0, beta=0, gamma=0)
trainer = pl.Trainer(accelerator="gpu", precision=16, max_epochs=10, overfit_batches=1, )