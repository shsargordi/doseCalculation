import torch,os
from torch import nn
import lightning as L
import numpy as np
import csv
import warnings
from monai.networks.nets import SwinUNETR
from .DVHMetric import DVHGlobalMetric
from .lr_scheduler import LinearWarmupCosineAnnealingLR

from monai.networks.nets import BasicUNetPlusPlus



warnings.filterwarnings("ignore")

class Dose_Prediction_Model(L.LightningModule):
    def __init__(self, args, Criterion):
        super().__init__()
        self.save_hyperparameters()

        self.args = args
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.batch_size = args.batch_size
        self.StructureNames = [
            "Brainstem", "Spinal cord", "Right parotid", "Left parotid",
            "Esophagus", "Larynx", "Mandible", "PTV56", "PTV63", "PTV70"
        ]

        # Model
        # self.DosePredictor = SwinUNETR(
        #     img_size=(128, 128, 128),
        #     in_channels=11,
        #     out_channels=1,
        #     feature_size=48,
        #     drop_rate=0.0,
        #     attn_drop_rate=0.0,
        #     dropout_path_rate=0.0,
        #     use_checkpoint=True,
        #     use_v2=True,
        # )

        self.DosePredictor = BasicUNetPlusPlus(
            spatial_dims=3,
            in_channels=11,
            out_channels=1,
            features=(32,64, 128, 256, 512,32),
            deep_supervision=False,
            norm=('instance', {'affine': True}),
            act=('LeakyReLU', {'inplace': True})
        )
       
        # Loss and Metrics
        self.criterion = Criterion
        self.Val_Metric = DVHGlobalMetric()
        self.Test_Metric = DVHGlobalMetric()

    def load_weights(self,weights='/store/Work/Gautier/SynthRad/DosePrediction/open-kbp/Logger/SwinUNETR_DoseGen/version_6/checkpoints/epoch=59-step=6000.ckpt'):
        # Load pre-trained weights
        model_dict = torch.load(weights)['net']
        self.DosePredictor.load_state_dict(model_dict)
        print('Loaded pretrained weights')

    def forward(self, x):
        d_pred = self.DosePredictor(x)

        if isinstance(d_pred, list):
            d_pred = d_pred[0]

        return d_pred

    def compute_loss(self, prediction, target, mask):

        if isinstance(prediction, list) :
            #For deep supervision 
            # Calculate loss for each output, weighted
            supervision_weights = [0.4, 0.3, 0.2, 0.1]
            losses = [self.criterion(pred,target,mask) * weight for pred, weight in zip(prediction, supervision_weights)]
            total_loss = sum(losses)
            return total_loss

        else :
            return self.criterion(prediction,target,mask)


    def training_step(self, batch, batch_idx):
        input = torch.cat((batch['ct'], batch['structure_masks']), dim=1)
        output = self.forward(input)
        loss = self.compute_loss(output, batch['dose'], batch['possible_dose_mask'])
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input = torch.cat((batch['ct'], batch['structure_masks']), dim=1)
        output = self.forward(input)
        val_loss = self.compute_loss(output, batch['dose'], batch['possible_dose_mask'])
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.args.batch_size)
        self.Val_Metric.update(
            pred_dose=output,
            gt_dose=batch["dose"],
            possible_dose_mask=batch["possible_dose_mask"],
            structure_masks=batch["structure_masks"],
            structure_names=self.StructureNames,
            voxel_dimensions=batch["voxel_dimensions"],
        )

    def test_step(self, batch, batch_idx):
        input = torch.cat((batch['ct'], batch['structure_masks']), dim=1)
        output = self.forward(input)
        test_loss = self.compute_loss(output, batch['dose'], batch['possible_dose_mask'])
        self.log('test_loss', test_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.args.batch_size)
        
        self.Test_Metric.update(
            pred_dose=output,
            gt_dose=batch["dose"],
            possible_dose_mask=batch["possible_dose_mask"],
            structure_masks=batch["structure_masks"],
            structure_names=self.StructureNames,
            voxel_dimensions=batch["voxel_dimensions"],
        )

        self.log_patient_scores(batch['patient_id'], output, batch)


    def on_validation_epoch_end(self, phase="val"):
        
        scores = self.Val_Metric.compute()
        scores = {f"{phase}_DVH_score": scores["dvh_score"].cpu(),f"{phase}_dose_score": scores["dose_score"].cpu()}
        self.log_dict(scores)
        self.Val_Metric.reset()

    def on_test_epoch_end(self,phase="test"):
        scores = self.Test_Metric.compute()
        scores = {f"{phase}_DVH_score": scores["dvh_score"].cpu(),f"{phase}_dose_score": scores["dose_score"].cpu()}
        self.log_dict(scores)
        self.Test_Metric.reset()

    def log_patient_scores(self, patient_ids, output, batch):
        voxel_dimensions = batch["voxel_dimensions"]
        gt = batch["dose"]
        mask = batch["possible_dose_mask"]
        structures = batch["structure_masks"]
        metric = DVHGlobalMetric()
        out_path = f'{self.args.output_path}.csv'
        header_written = hasattr(self, "_header_written")
        if not header_written:
            if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
                with open(out_path, "w", newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["patient_id", "dvh_score", "dose_score"])
            self._header_written = True

        with open(out_path, "a", newline='') as f:
            writer = csv.writer(f)
            for i, pid in enumerate(patient_ids):
                metric.reset()
                metric.update(
                    pred_dose=output[i:i+1],
                    gt_dose=gt[i:i+1],
                    possible_dose_mask=mask[i:i+1],
                    structure_masks=structures[i:i+1],
                    structure_names=self.StructureNames,
                    voxel_dimensions=voxel_dimensions[i:i+1],
                )
                result = metric.compute()
                writer.writerow([pid, result["dvh_score"].item(), result["dose_score"].item()])

    def configure_optimizers(self):
        if self.args.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay, momentum=self.args.momentum)
        else:
            raise ValueError("Unsupported optimizer type")

        if self.args.use_scheduler:

            scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=self.args.warmup_epochs, max_epochs=self.args.max_epochs)
            scheduler = {
                "scheduler": scheduler,
                "name": "lr_history",
                "interval": "step",
            }

            return [optimizer], [scheduler]
        else:
            return optimizer