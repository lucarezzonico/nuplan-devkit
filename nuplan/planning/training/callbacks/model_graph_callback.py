from pathlib import Path
from typing import Optional

import pytorch_lightning as pl


class ModelGraphAtEpochEnd(pl.callbacks.Callback):
    """Customized callback for saving Lightning checkpoint for every epoch."""

    def __init__(
        self,
        dirpath: Optional[str] = None,
    ):
        """
        Initialize the callback
        :param dirpath: Directory where the checkpoints are saved.
        """
        super().__init__(dirpath=dirpath)

    def on_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Customized callback function to save checkpoint every epoch.
        :param trainer: Pytorch lightning trainer instance.
        :param pl_module: LightningModule.
        """
        checkpoint_dir = Path(trainer.model_graph_callback.dirpath).parent / 'checkpoints'
        checkpoint_name = f'epoch={trainer.current_epoch}.ckpt'
        checkpoint_path = checkpoint_dir / checkpoint_name
        trainer.save_checkpoint(str(checkpoint_path))
