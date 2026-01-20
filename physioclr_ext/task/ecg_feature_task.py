from dataclasses import dataclass, field
from fairseq_signals.tasks import (
    AudioPretrainingTask, AudioPretrainingConfig, register_task
)
from physioclr_ext.data.ecg_feature_dataset import ECGFeatDataset


@dataclass
class ECGFeatureTaskConfig(AudioPretrainingConfig):
    data: str = field(default="", metadata={"help": "txt list of .npy files"})


@register_task("ecg_feature", dataclass=ECGFeatureTaskConfig)
class ECGFeatureTask(AudioPretrainingTask):
    """
    Minimal wrapper so Hydra can load ECGFeatDataset for PhysioCLR pre-training.
    """

    def load_dataset(self, split: str, **kwargs):
        if split not in self.datasets:
            with open(self.cfg.data) as f:
                paths = [ln.strip() for ln in f if ln.strip()]
            self.datasets[split] = ECGFeatDataset(paths)
