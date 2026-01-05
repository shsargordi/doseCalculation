import pandas as pd
import numpy as np
import SimpleITK as sitk
import os 
import os
import nibabel as nib
from glob import glob
from monai.transforms import *
from Modules.model import *
import tqdm
from Modules.DVHMetric import *
from Modules.TTA import test_time_augmented_inference

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def select_ctv_variant(structlist, base):
    """
    Select the most suitable variant for a given CTV base name (CTV_1, CTV_2, etc.)
    Priority: exact match > highest dose suffix > longest name
    """
    candidates = [s for s in structlist if s.startswith(base)]
    if not candidates:
        return None

    # Prioritize exact match
    exact = [s for s in candidates if s == f"{base}.nii.gz"]
    if exact:
        return exact[0]

    # Else sort by preferred suffix
    # Example scoring: 70 > 66 > rest
    def score(name):
        if '70' in name:
            return 3
        elif '66' in name:
            return 2
        elif any(c in name for c in ['A', 'B']):
            return 1
        return 0

    candidates.sort(key=score, reverse=True)
    return candidates[0] if candidates else None


out_path = 'output10cases.csv'
out_path_dvh = 'output10casesDVH.csv'
#######
#Defining how we load the data
#######

StructureNames = ["Canal medullaire", "Tronc", "Parotide D", "Parotide G",
                  "Oesophage", "Larynx", "Mandibule"]


pnbdict = {}

add = '/path/to/data/'

for pid in os.listdir(add):
    try:
        base_path = f'/path/to/data/{pid}'
        rtstruct_path = os.path.join(base_path, 'RTStruct')
        structlist = os.listdir(rtstruct_path)


        # Select proper CTVs
        ctv1 = select_ctv_variant(structlist, "CTV 1")
        ctv2 = select_ctv_variant(structlist, "CTV 2")
        ctv3 = select_ctv_variant(structlist, "CTV 3")

        if not all([ctv1, ctv2, ctv3]):
            continue  # skip patient

        # Paths to the selected masks
        ctv_paths = [os.path.join(rtstruct_path, ctv) for ctv in [ctv3, ctv2, ctv1]]

        # Other structures
        struct_list = [os.path.join(rtstruct_path, f'{s}.nii.gz') for s in StructureNames]
        check_struct = np.all([os.path.isfile(x) for x in struct_list])


        ct_synth = f'/path/to/dataSynth/{pid}.nii.gz'

        ct = os.path.join(base_path, 'ct.nii.gz')
        dose = os.path.join(base_path, 'dosi.nii.gz')
        mask = os.path.join(base_path, 'mask.nii.gz')

        #check_images =  np.all([os.path.isfile(x) for x in [ct_synth,cbct,ct,mask]])
        check_images =  np.all([os.path.isfile(x) for x in [ct_synth,ct,mask]])


        if check_struct and check_images :

            
            pnbdict[pid] = {
                'ct': ct,
                #'cbct': cbct,
                'ct_synth': ct_synth,
                'dose': dose,
                'structure_masks': struct_list+ctv_paths,
                'possible_dose_mask':mask
            }

    except Exception as e:
        continue


#######
#Defining how we load the data (transforms from monai)
#######

#all_keys = ['ct','cbct','structure_masks','possible_dose_mask','ct_synth']
all_keys = ['ct','structure_masks','possible_dose_mask','ct_synth']
#image_key = ['ct','cbct','ct_synth']
image_key = ['ct','ct_synth']
mask_keys = ['structure_masks','possible_dose_mask']

Transforms = Compose([ LoadImaged(keys=all_keys),
            EnsureChannelFirstd(keys=all_keys),
            ResampleToMatchd(keys=['structure_masks','possible_dose_mask','ct_synth'],key_dst='ct'),
            #ResampleToMatchd(keys=['cbct','structure_masks','possible_dose_mask','ct_synth'],key_dst='ct'),
            ToTensord(keys=all_keys),
            OrientationD(keys=all_keys,axcodes='SPL'),
            ScaleIntensityRanged(keys=image_key, a_min=-1024, a_max=1500, b_min=-1.024, b_max=1.5, clip=True),
            Resized(keys=image_key,spatial_size=(256,256,79),mode='bilinear'),
            Resized(keys=mask_keys,spatial_size=(256,256,79),mode='nearest-exact'),
            EnsureTyped(keys=all_keys),
        ])


#Load Model
weights = '/path/to/doseCal/weight/epoch=399-step=40000.ckpt'
model = Dose_Prediction_Model.load_from_checkpoint(weights)
model = model.to('cuda')

#Loop
predictions = {}

for key in tqdm.tqdm(pnbdict.keys()):

    try:
    
        x = pnbdict[key]
        #print(key)
        x = Transforms(x)
        with torch.no_grad():
            model.eval()

            #test with TTA
            #outCBCT_TTA = test_time_augmented_inference(model, x, device='cuda',n_intensity_aug=10,imgkey='cbct').to('cuda')
            outCT_TTA = test_time_augmented_inference(model, x, device='cuda',n_intensity_aug=10,imgkey='ct').to('cuda')
            outSCT_TTA = test_time_augmented_inference(model, x, device='cuda',n_intensity_aug=10,imgkey='ct_synth').to('cuda')
            gt = outCT_TTA[0][0].cpu().numpy()
            #predCBCT_TTA = outCBCT_TTA[0][0].cpu().numpy()
            predSCT_TTA = outSCT_TTA[0][0].cpu().numpy()
            
            # #test without TTA
            # input = torch.cat((x['cbct'], x['structure_masks']), dim=0).unsqueeze(0).to('cuda')
            # out = model.forward(input).to('cuda')
            # outCBCT = out[0]*x['possible_dose_mask'].to('cuda')
            # predCBCT =outCBCT[0].cpu().numpy()

            input = torch.cat((x['ct_synth'], x['structure_masks']), dim=0).unsqueeze(0).to('cuda')
            out = model.forward(input).to('cuda')
            outSCT = out[0]*x['possible_dose_mask'].to('cuda')
            predSCT =outSCT[0].cpu().numpy()
       
        # Extract arrays from tensors
        mask = x['possible_dose_mask'] # (1, H, W, D)

        #Dose score
        # #scoreCBCT = np.sum(np.abs(gt - predCBCT)) / np.sum(mask)
        # #scoreCBCT_TTA = np.sum(np.abs(gt - predCBCT_TTA)) / np.sum(mask)
        scoreSCT = np.sum(np.abs(gt - predSCT)) / np.sum(mask)
        scoreSCT_TTA = np.sum(np.abs(gt - predSCT_TTA)) / np.sum(mask)


        #DVH score
        #dvh_score_CBCT = get_dvh_diff_single_case(x,gt,predCBCT)
        #dvh_score_CBCT_TTA = get_dvh_diff_single_case(x,gt,predCBCT_TTA)
        dvh_score_SCT = get_dvh_diff_single_case(x,gt,predSCT)
        dvh_score_SCT_TTA = get_dvh_diff_single_case(x,gt,predSCT_TTA)


        #DVH metrics
        #dvh_metrics_CBCT = get_dvh_diff_metrics_single_case(x,gt,predCBCT)
        #dvh_metrics_CBCT['modality'] = 'CBCT'
        #dvh_metrics_CBCT_TTA = get_dvh_diff_metrics_single_case(x,gt,predCBCT_TTA)
        #dvh_metrics_CBCT_TTA['modality'] = 'CBCT_TTA'
        dvh_metrics_SCT = get_dvh_diff_metrics_single_case(x,gt,predSCT)
        dvh_metrics_SCT['modality'] = 'SCT'
        dvh_metrics_SCT_TTA = get_dvh_diff_metrics_single_case(x,gt,predSCT_TTA)
        dvh_metrics_SCT_TTA['modality'] = 'SCT_TTA'
        

        with open(out_path, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([key,scoreSCT,scoreSCT_TTA,dvh_score_SCT,dvh_score_SCT_TTA])

            #writer.writerow([key,scoreCBCT,scoreCBCT_TTA,scoreSCT,scoreSCT_TTA,dvh_score_CBCT,dvh_score_CBCT_TTA,dvh_score_SCT,dvh_score_SCT_TTA])

        with open(out_path_dvh, "a", newline='') as f:
            writer = csv.writer(f)

            #for metrics in [dvh_metrics_CBCT,dvh_metrics_CBCT_TTA,dvh_metrics_SCT,dvh_metrics_SCT_TTA]:
            for metrics in [dvh_metrics_SCT,dvh_metrics_SCT_TTA]:
                metricsrow = [key]+[metrics[key] for key in metrics.keys()]
                writer.writerow(metricsrow)


    except Exception as e:

        print(e)
        
        

