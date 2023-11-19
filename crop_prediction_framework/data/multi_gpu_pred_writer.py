# Taken and adapted from: https://github.com/Lightning-AI/lightning/discussions/9259

import logging
import os, os.path
import shutil
import tempfile
import pickle
import itertools

import torch
from pytorch_lightning.callbacks import BasePredictionWriter
from torch.nn.functional import softmax
import numpy as np

logger = logging.getLogger(__name__)

class CustomWriter(BasePredictionWriter):
    """Pytorch Lightning Callback that saves predictions and the corresponding batch
    indices in a temporary folder when using multigpu inference.

    Args:
        write_interval (str): When to perform write operations. Defaults to 'epoch'
    """

    def __init__(self, write_interval, output_file) -> None:
        self.output_file=output_file
        super().__init__(write_interval)

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        """Saves predictions after running inference on all samples."""

        # We need to save predictions in the most secure manner possible to avoid
        # multiple users and processes writing to the same folder.
        # For that we will create a tmp folder that will be shared only across
        # the DDP processes that were created
        if trainer.is_global_zero:
            output_dir = [
                tempfile.mkdtemp(),
            ]
            logger.info(
                "Created temporary folder to store predictions: {}.".format(
                    output_dir[0]
                )
            )
        else:
            output_dir = [
                None,
            ]

        torch.distributed.broadcast_object_list(output_dir)

        # Make sure every process received the output_dir from RANK=0
        torch.distributed.barrier()  
        # Now that we have a single output_dir shared across processes we can save
        # prediction along with their indices.
        self.output_dir = output_dir[0]
        # this will create N (num processes) files in `output_dir` each containing
        # the predictions of it's respective rank
        torch.save(
            predictions, os.path.join(self.output_dir, f"pred_{trainer.global_rank}.pt")
        )
        # optionally, you can also save `batch_indices` to get the information about
        # the data index from your prediction data
        torch.save(
            batch_indices,
            os.path.join(self.output_dir, f"batch_indices_{trainer.global_rank}.pt"),
        )

        torch.distributed.barrier() 
        
        if trainer.is_global_zero:
            self.gather_all_predictions()
            self.cleanup()

    def gather_all_predictions(self):
        """Reads all saved predictions from the self.output_dir into one single
        Prediciton object respecting the original order of the samples.
        """
        files = sorted(os.listdir(self.output_dir))

        outputs = list(itertools.chain.from_iterable([torch.load(os.path.join(self.output_dir, f)) for f in files if "pred" in f]))
        indices = list(itertools.chain.from_iterable([torch.load(os.path.join(self.output_dir, f))[0] for f in files if "batch_indices" in f]))

        output_dict = {}
        output_dict['probabilities'] = np.concatenate([softmax(x[0].float(), dim=-1).numpy() for x in outputs])
        output_dict['labels'] = np.concatenate([x[1].numpy() for x in outputs])
        output_dict['predictions'] = np.argmax(output_dict['probabilities'], axis=-1)
        output_dict['idx'] = np.concatenate([np.array(x) for x in indices])

        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        with open(self.output_file, 'wb') as f:
            pickle.dump(output_dict, f)

    def cleanup(self):
        """Cleans temporary files."""
        logger.info("Cleanup temporary folder: {}.".format(self.output_dir))
        shutil.rmtree(self.output_dir)