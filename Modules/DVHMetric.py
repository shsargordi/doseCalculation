import torch
from torchmetrics import Metric
from typing import Dict, List, Optional
import numpy as np


class DVHGlobalMetric(Metric):
    """
    DVH and dose evaluation metric for dose prediction models in PyTorch Lightning.
    Automatically sets clinical DVH criteria from standard OpenKBP structures.
    """

    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("total_dose_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_samples", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_dvh_error", default=torch.tensor(0.0), dist_reduce_fx="sum")

        # Predefined structure-metric mapping based on clinical guidelines
        self.all_dvh_metrics: Dict[str, List[str]] = {
            "Brainstem": ["D_0.1_cc"],
            "Spinal cord": ["D_0.1_cc"],
            "Right parotid": ["mean"],
            "Left parotid": ["mean"],
            "Esophagus": ["mean"],
            "Larynx": ["mean"],
            "Mandible": ["D_0.1_cc"],
            "PTV56": ["D_99"],
            "PTV63": ["D_99"],
            "PTV70": ["D_99"],
        }

    def update(
        self,
        pred_dose: torch.Tensor,           # (B, 1, H, W, D)
        gt_dose: torch.Tensor,             # (B, 1, H, W, D)
        possible_dose_mask: torch.Tensor,  # (B, 1, H, W, D)
        structure_masks: torch.Tensor,     # (B, N_rois, H, W, D)
        structure_names: List[str],        # length N_rois
        voxel_dimensions: torch.Tensor     # (B, 3): spacingH, spacingW, spacingD
    ):
        if isinstance(pred_dose, list) :
            pred_dose = pred_dose[0]

        b = pred_dose.shape[0]

        name_to_index = {name: idx for idx, name in enumerate(structure_names)}

        for i in range(b):
            spacing = voxel_dimensions[i]  # (3,)
            voxel_volume = torch.prod(spacing).item()  # mm^3

            pd = pred_dose[i, 0].cpu().numpy().flatten()
            gt = gt_dose[i, 0].cpu().numpy().flatten()
            mask = possible_dose_mask[i, 0].cpu().numpy().flatten()
            pd = pd * mask
            dose_error = np.sum(np.abs(pd - gt)) / np.sum(mask)
            self.total_dose_error += torch.tensor(dose_error, device=self.device)

            dvh_error = 0.0
            for roi_name, metrics in self.all_dvh_metrics.items():
                if roi_name not in name_to_index:
                    continue  # skip missing structures

                roi_idx = name_to_index[roi_name]
                roi_mask = structure_masks[i, roi_idx].cpu().numpy().flatten().astype(bool)
                if not roi_mask.any():
                    continue # skip non annotated structures

                pd_roi = pd[roi_mask]
                gt_roi = gt[roi_mask]

                for metric in metrics:
                    pd_val = self._compute_metric(metric, pd_roi, voxel_volume)
                    gt_val = self._compute_metric(metric, gt_roi, voxel_volume)
                    dvh_error += abs(pd_val - gt_val)

            self.total_dvh_error += torch.tensor(dvh_error, device=self.device)
            self.num_samples += 1

    def _compute_metric(self, metric: str, dose: np.ndarray, voxel_volume: float) -> float:
        if dose.size == 0:
            return 0.0

        if metric == "D_0.1_cc":
            cc_volume = 0.1 * 1000  # mm³
            voxels = max(1, round(cc_volume / voxel_volume))
            frac = 100 - voxels / len(dose) * 100
            return np.percentile(dose, frac)
        elif metric == "mean":
            return dose.mean()
        elif metric == "D_99":
            return np.percentile(dose, 1)
        elif metric == "D_95":
            return np.percentile(dose, 5)
        elif metric == "D_1":
            return np.percentile(dose, 99)
        else:
            raise ValueError(f"Unknown DVH metric: {metric}")

    def compute(self):
        dose_score = self.total_dose_error / self.num_samples
        dvh_score = self.total_dvh_error / self.num_samples
        return {
            "dose_score": dose_score,
            "dvh_score": dvh_score,
        }

    def reset(self):
        self.total_dose_error = torch.tensor(0.0, device=self.device)
        self.total_dvh_error = torch.tensor(0.0, device=self.device)
        self.num_samples = torch.tensor(0, device=self.device)

def compute_dvh_metrics(x, dose):
    """
    Compute DVH metrics for a single patient, given dose and structure masks.
    Returns a dict with keys as (metric, ROI) and values as the computed metric.
    """
    
    
    # Define DVH metrics per structure
    ALL_DVH_METRICS = {
        "Canal_medullaire": ["D_0.1_cc"],
        "Tronc": ["D_0.1_cc"],
        "Parotide_D": ["mean"],
        "Parotide_G": ["mean"],
        "Oesophage": ["mean"],
        "Larynx": ["mean"],
        "Mandibule": ["D_0.1_cc"],
        "PTV56": ["D_99"],
        "PTV63": ["D_99"],
        "PTV70": ["D_99"],
    }

    FULL_ROI_LIST = list(ALL_DVH_METRICS.keys())
    voxel_volume_mm3  = 2 ### ( OUR DATA IS RESAMPLED TO 1x1x2 mm) ## ADAPT THIS TO YOUR CODE
    voxels_within_tenths_cc = np.maximum(1, np.round(100 / voxel_volume_mm3))
    structures = x['structure_masks'].squeeze().cpu().bool().numpy()

    metrics_result = {}

    for i, roi in enumerate(FULL_ROI_LIST):
        roi_mask = structures[i]
        if roi_mask is None or not np.any(roi_mask):
            print(f"Skipping {roi} (empty or None)")
            continue

        roi_dose = dose[roi_mask]

        for metric in ALL_DVH_METRICS[roi]:
            if metric == "D_0.1_cc":
                roi_size = len(roi_dose)
                frac_vol = 100 - voxels_within_tenths_cc / roi_size * 100
                value = np.percentile(roi_dose, frac_vol)
            elif metric == "mean":
                value = roi_dose.mean()
            elif metric == "D_99":
                value = np.percentile(roi_dose, 1)
            elif metric == "D_95":
                value = np.percentile(roi_dose, 5)
            elif metric == "D_1":
                value = np.percentile(roi_dose, 99)
            else:
                raise ValueError(f"Unsupported metric: {metric}")
            
            metrics_result[(metric, roi)] = value

    return metrics_result


def compute_dvh_score(reference_metrics, predicted_metrics):
    """
    Compute average absolute error between predicted and reference DVH metrics.
    """
    all_keys = reference_metrics.keys()
    errors = [abs(reference_metrics[k] - predicted_metrics[k]) for k in all_keys if k in predicted_metrics]
    return np.nanmean(errors)


def get_dvh_diff_single_case(x, pred_dose, gt_dose):
    """
    Compute DVH difference score for a single pair of predicted and ground-truth dose distributions.
    """
    ref_metrics = compute_dvh_metrics(x, gt_dose)
    pred_metrics = compute_dvh_metrics(x, pred_dose)
    return compute_dvh_score(ref_metrics, pred_metrics)


def get_dvh_diff_metrics_single_case(x, pred_dose, gt_dose):
    """
    Compute DVH difference (delta) for each metric between predicted and ground-truth doses.
    Returns a dictionary: keys are "metric_structure", values are delta or empty string if missing.
    """
    ref_metrics = compute_dvh_metrics(x, gt_dose)
    pred_metrics = compute_dvh_metrics(x, pred_dose)

    all_keys = [
        ('D_0.1_cc', 'Canal_medullaire'),
        ('D_0.1_cc', 'Tronc'),
        ('mean', 'Parotide_D'),
        ('mean', 'Parotide_G'),
        ('mean', 'Esophagus'),
        ('mean', 'Larynx'),
        ('D_0.1_cc', 'Mandibule'),
        ('D_99', 'PTV56'),
        ('D_99', 'PTV63'),
        ('D_99', 'PTV70')
    ]

    delta_results = {}
    for k in all_keys:
        key_name = f"{k[0]}_{k[1]}"
        if k in ref_metrics and k in pred_metrics:
            delta_results[key_name] = abs(ref_metrics[k] - pred_metrics[k])
        else:
            delta_results[key_name] = ""  # Leave empty if structure is missing

    return delta_results