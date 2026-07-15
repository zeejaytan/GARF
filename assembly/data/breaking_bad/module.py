from typing import Literal, Optional, List, Dict

import lightning as L
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from . import BreakingBadUniform, BreakingBadWeighted


class BreakingBadDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_root: str = "data",
        categories: List[str] = ["everyday"],
        min_parts: int = 2,
        max_parts: int = 20,
        num_points_to_sample: int = 1000,
        min_points_per_part: int = 20,
        sample_method: Literal["uniform", "weighted"] = "uniform",
        batch_size: int = 32,
        num_workers: int = 16,
        num_removal: int = 0,
        num_redundancy: int = 0,
        multi_ref: bool = False,
        mesh_sample_strategy: Literal["uniform", "poisson"] = "poisson",
        random_anchor: bool = False,
        rim_oversample_frac: float = 0.0,
        rim_band_frac: float = 0.05,
        rim_relief_pct: float = 85.0,
        frac_erode_prob: float = 0.0,
        frac_erode_min: float = 0.3,
        frac_erode_max: float = 1.0,
        additional_data_root: Optional[Dict[str, str]] = None,
    ):
        super().__init__()
        self.data_root = data_root
        self.categories = categories
        self.min_parts = min_parts
        self.max_parts = max_parts
        self.num_points_to_sample = num_points_to_sample
        self.min_points_per_part = min_points_per_part
        self.sample_method = sample_method
        self.batch_size = batch_size
        self.num_workers = num_workers
        # Please be noted that num_removal and num_redundancy are only used in the testing phase
        self.num_removal = num_removal
        self.num_redundancy = num_redundancy

        # Please be noted that multi_ref is only used in the training phase
        self.multi_ref = multi_ref

        self.mesh_sample_strategy = mesh_sample_strategy
        self.random_anchor = random_anchor

        # Fracture-rim oversampling remedy (0.0 = original sampling behaviour).
        self.rim_oversample_frac = rim_oversample_frac
        self.rim_band_frac = rim_band_frac
        self.rim_relief_pct = rim_relief_pct
        self.frac_erode_prob = frac_erode_prob
        self.frac_erode_min = frac_erode_min
        self.frac_erode_max = frac_erode_max

        print("Using mesh sample strategy:", self.mesh_sample_strategy)
        if self.frac_erode_prob > 0.0:
            print(
                f"Worn-break augmentation ON (train): prob={self.frac_erode_prob}, "
                f"strength in [{self.frac_erode_min}, {self.frac_erode_max}]"
            )
        if self.rim_oversample_frac > 0.0:
            print(
                f"Fracture-rim oversampling ON: frac={self.rim_oversample_frac}, "
                f"band_frac={self.rim_band_frac}, relief_pct={self.rim_relief_pct}"
            )

        # If breaking_bad_other_data_root is provided
        self.additional_data_root = additional_data_root

        if self.sample_method == "uniform":
            self.dataset_cls = BreakingBadUniform
        elif self.sample_method == "weighted":
            self.dataset_cls = BreakingBadWeighted
        else:
            raise ValueError(f"Invalid sample method: {self.sample_method}")

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None

    def setup(self, stage):
        if stage == "fit":

            self.train_dataset = ConcatDataset(
                [
                    self.dataset_cls(
                        split="train",
                        data_root=(
                            self.additional_data_root[category]
                            if self.additional_data_root is not None
                            and category in self.additional_data_root
                            else self.data_root
                        ),
                        category=category,
                        min_parts=self.min_parts,
                        max_parts=self.max_parts,
                        num_points_to_sample=self.num_points_to_sample,
                        min_points_per_part=self.min_points_per_part,
                        multi_ref=self.multi_ref,
                        mesh_sample_strategy=self.mesh_sample_strategy,
                        random_anchor=self.random_anchor,
                        rim_oversample_frac=self.rim_oversample_frac,
                        rim_band_frac=self.rim_band_frac,
                        rim_relief_pct=self.rim_relief_pct,
                        frac_erode_prob=self.frac_erode_prob,
                        frac_erode_min=self.frac_erode_min,
                        frac_erode_max=self.frac_erode_max,
                    )
                    for category in self.categories
                ]
            )

            self.val_dataset = ConcatDataset(
                [
                    self.dataset_cls(
                        split="val",
                        data_root=(
                            self.additional_data_root[category]
                            if self.additional_data_root is not None
                            and category in self.additional_data_root
                            else self.data_root
                        ),
                        category=category,
                        min_parts=self.min_parts,
                        max_parts=self.max_parts,
                        num_points_to_sample=self.num_points_to_sample,
                        min_points_per_part=self.min_points_per_part,
                        mesh_sample_strategy=self.mesh_sample_strategy,
                        random_anchor=self.random_anchor,
                        rim_oversample_frac=self.rim_oversample_frac,
                        rim_band_frac=self.rim_band_frac,
                        rim_relief_pct=self.rim_relief_pct,
                    )
                    for category in self.categories
                ]
            )

        if stage == "test" or stage == "predict":
            self.val_dataset = ConcatDataset(
                [
                    self.dataset_cls(
                        split="val",
                        data_root=(
                            self.additional_data_root[category]
                            if self.additional_data_root is not None
                            and category in self.additional_data_root
                            else self.data_root
                        ),
                        category=category,
                        min_parts=self.min_parts,
                        max_parts=self.max_parts,
                        num_points_to_sample=self.num_points_to_sample,
                        min_points_per_part=self.min_points_per_part,
                        num_removal=self.num_removal,
                        num_redundancy=self.num_redundancy,
                        mesh_sample_strategy=self.mesh_sample_strategy,
                        random_anchor=self.random_anchor,
                        rim_oversample_frac=self.rim_oversample_frac,
                        rim_band_frac=self.rim_band_frac,
                        rim_relief_pct=self.rim_relief_pct,
                    )
                    for category in self.categories
                ]
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            persistent_workers=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=False,
            collate_fn=BreakingBadWeighted.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=BreakingBadWeighted.collate_fn
        )
