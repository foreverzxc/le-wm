import numpy as np
import torch
from stable_pretraining import data as dt
from lightning.pytorch.callbacks import Callback

def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


class ZScoreNormalizer:
    """Picklable z-score normalizer — uses a class instead of a closure so it
    survives pickle when DataLoader workers are spawned (required by LanceDataset)."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return ((x - self.mean) / self.std).float()


def get_column_normalizer(dataset, source: str, target: str):
    """Get normalizer for a specific column in the dataset.

    For large datasets (e.g. LIBERO), samples episodes instead of scanning
    all data to avoid minutes of HDF5 I/O during startup.
    """
    # Use load_episode sampling path when available (LiberoDataset etc.)
    n_episodes = len(dataset._episode_meta) if hasattr(dataset, '_episode_meta') else 0
    if n_episodes > 10 and hasattr(dataset, 'load_episode'):
        # Sample episodes evenly across the dataset
        rng = np.random.default_rng(42)
        n_sample = min(10, n_episodes)
        indices = np.linspace(0, n_episodes - 1, n_sample, dtype=int)
        parts = [dataset.load_episode(int(i))[source] for i in indices]
        col_data = np.concatenate(parts, axis=0)
    else:
        col_data = dataset.get_col_data(source)

    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()
    return dt.transforms.WrapTorchTransform(ZScoreNormalizer(mean, std), source=source, target=target)

class SaveCkptCallback(Callback):
    """Callback to save model checkpoint after each epoch using save_pretrained."""

    def __init__(self, run_name, cfg, epoch_interval: int = 1):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._save(pl_module.model, trainer.current_epoch + 1)

            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._save(pl_module.model, trainer.current_epoch + 1)

    def _save(self, model, epoch):
        from pathlib import Path
        import json
        import torch
        from omegaconf import OmegaConf
        from stable_worldmodel.data.utils import get_cache_dir

        run_dir = Path(get_cache_dir(), "checkpoints")
        run_dir.mkdir(parents=True, exist_ok=True)

        weight_path = run_dir / f"{self.run_name}_weights_epoch_{epoch}.pt"
        torch.save(model.state_dict(), weight_path)

        config_path = run_dir / f"{self.run_name}_config_epoch_{epoch}.json"
        config_dict = OmegaConf.to_container(self.cfg, resolve=True)
        config_path.write_text(json.dumps(config_dict, indent=2, default=str))