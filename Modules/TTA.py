import torch
import numpy as np
import random

def apply_flip(x, axes):
    return {k: torch.flip(v, dims=axes) for k, v in x.items() if k in ['ct', 'structure_masks']}

def reverse_flip(x, axes):
    return torch.flip(x, dims=axes)

def apply_intensity_aug(x):
    x_aug = x.copy()

    ct = x[f'ct'].clone()

    # Gamma correction (random gamma between 0.7 and 1.5)
    if random.random() < 0.5:
        gamma = random.uniform(0.7, 1.5)
        ct = ct.clamp_min(0) ** gamma

    # Gaussian noise (std between 0 and 0.05 * max)
    if random.random() < 0.5:
        std = ct.max() * 0.05
        noise = torch.randn_like(ct) * std
        ct = ct + noise

    # Intensity scaling (like brightness shift)
    if random.random() < 0.5:
        factor = random.uniform(0.9, 1.1)
        ct = ct * factor

    x_aug['ct'] = ct
    return x_aug

def test_time_augmented_inference(model, x, device='cuda',imgkey='ct', n_intensity_aug=10):
    model.eval()
    with torch.no_grad():
        base_input = {
            'ct': x[f'{imgkey}'].unsqueeze(0).to(device),
            'structure_masks': x['structure_masks'].unsqueeze(0).to(device),
            'possible_dose_mask': x['possible_dose_mask'].unsqueeze(0).to(device)
        }

        preds = []

        # Original + flips
        flip_axes_list = [
            [], [2], [3], [4],
            [2, 3], [2, 4], [3, 4],
            [2, 3, 4]
        ]

        for axes in flip_axes_list:
            # Flip inputs
            if axes:
                aug = apply_flip(base_input, axes)
            else:
                aug = base_input.copy()

            # Optionally apply intensity augmentation
            aug = apply_intensity_aug(aug)

            # Prepare input
            inp = torch.cat((aug['ct'], aug['structure_masks']), dim=1)
            out = model(inp)

            # Reverse flips
            if axes:
                out = reverse_flip(out, axes)

            # Apply possible dose mask
            out = out * base_input['possible_dose_mask']
            preds.append(out)

        # Add extra randomized intensity-only variants (no flips)
        for _ in range(n_intensity_aug):
            aug = apply_intensity_aug(base_input.copy())
            inp = torch.cat((aug['ct'], aug['structure_masks']), dim=1)
            out = model(inp)
            out = out * base_input['possible_dose_mask']
            preds.append(out)

        # Average predictions
        final_pred = torch.stack(preds, dim=0).mean(dim=0)

    return final_pred