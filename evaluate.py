import os
import numpy as np
import json
from tqdm import tqdm
import pandas as pd


#################################################
# ---------------------- User Config ------------------------
#################################################

# Where your predictions are stored
prediction_folder = './PredResults/dosesp'   # e.g. '../PredResults' or '../results_mednext'

# CSV output path
csv_path = './eval_results.csv'

# Choose mode: "overwrite_all", "append", "overwrite_row"
csv_mode = "append"

# Optional: manually set model name. If None, infer from prediction_folder.
model_name_override = 'dosesp'

decimal_places = 3

#################################################
# ---------------------- Data Loading ------------------------
#################################################

df = pd.read_csv('meta_files/meta_data.csv')
df = df.loc[df['phase'] == 'valid']

npz_paths = df['npz_path'].tolist()
evaluation_list = [path.split('/')[-1].split('.')[0] for path in npz_paths]

site_list = df['site'].tolist()
PTVHighname_list = ['PTVHighOPT' if site == 1 else 'PTV' for site in site_list]

scale_dose_Dict = json.load(open('meta_files/PTV_DICT.json'))


#################################################
# ---------------------- OAR Lists ------------------------
#################################################

HaN_OAR_LIST = [
    'Cochlea_L','Cochlea_R','Eyes','Lens_L','Lens_R',
    'OpticNerve_L','OpticNerve_R','Chiasim','LacrimalGlands',
    'BrachialPlexus','Brain','BrainStem_03','Esophagus',
    'Lips','Lungs','Trachea','Posterior_Neck','Shoulders',
    'Larynx-PTV','Mandible-PTV','OCavity-PTV','ParotidCon-PTV',
    'ParotidIps-PTV','Parotids-PTV','PharConst-PTV',
    'Submand-PTV','SubmandL-PTV','SubmandR-PTV',
    'Thyroid-PTV','SpinalCord_05'
]

Lung_OAR_LIST = [
    "PTV_Ring.3-2","Total Lung-GTV","SpinalCord",
    "Heart","LAD","Esophagus","BrachialPlexus",
    "GreatVessels","Trachea","Body_Ring0-3"
]


#################################################
# ---------------------- Metric Functions ------------------------
#################################################

def fmt_ms(m, s, nd):
    return f"{m:.{nd}f}+-{s:.{nd}f}"


def compute_D_metrics(dose, mask):
    """Compute common DVH points inside a ROI mask."""
    roi = dose[mask > 0]
    
    if roi.size == 0:
        # Return zeros to avoid crash; you can also choose to raise an error
        return {"D1":0.0,"D2":0.0,"D50":0.0,"D95":0.0,"D98":0.0,"D99":0.0,"Dmean":0.0}
        
    return {
        "D1": np.percentile(roi, 99),
        "D2": np.percentile(roi, 98),
        "D50": np.percentile(roi, 50),
        "D95": np.percentile(roi, 5),
        "D98": np.percentile(roi, 2),
        "D99": np.percentile(roi, 1),
        "Dmean": np.mean(roi)
    }


def compute_HI(dose, mask):
    """HI = (D2 - D98) / D50."""
    m = compute_D_metrics(dose, mask)
    return (m["D2"] - m["D98"]) / (m["D50"] + 1e-8)


def compute_CI(dose, ptv_mask, prescription):
    """Voxel-based CI using prescription isodose volume."""
    ptv = ptv_mask > 0
    pres_mask = dose >= prescription

    V_ptv = np.sum(ptv)
    V_pres = np.sum(pres_mask)
    V_overlap = np.sum(ptv & pres_mask)

    if V_ptv == 0 or V_pres == 0:
        return 0.0

    return (V_overlap ** 2) / (V_ptv * V_pres + 1e-8)


def compute_D01cc_voxel(dose, mask):
    """Approximate D0.1cc without spacing by top 0.1% voxels mean."""
    roi = dose[mask > 0]
    if len(roi) == 0:
        return 0.0
    k = max(1, int(0.001 * len(roi)))  # 0.1% voxel approximation
    return np.mean(np.sort(roi)[-k:])


def infer_model_name(pred_folder: str) -> str:
    """Infer a readable model name from the prediction folder path."""
    base = os.path.basename(os.path.normpath(pred_folder))
    if base == "" or base == "." or base == "..":
        base = pred_folder.replace("/", "_").replace("\\", "_")
    return base


def mean_std(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.mean(x)), float(np.std(x))


#################################################
# ---------------------- Main Evaluation ------------------------
#################################################

Metric1_MAE_list = []

DoseScore_list = []
DVHScore_list = []
HI_list = []
CI_list = []
D01cc_list = []
DmeanOAR_list = []

# NEW: PTV metric diffs
PTV_D1_list = []
PTV_D95_list = []
PTV_D99_list = []

for i in tqdm(range(len(evaluation_list))):

    plan_file_name = evaluation_list[i]
    PTVHighname = PTVHighname_list[i]

    patient_id = plan_file_name.split('+')[0]
    data_path = npz_paths[i]

    data_npz = np.load(data_path, allow_pickle=True)
    data_dict = dict(data_npz)['arr_0'].item()

    # ----- Dose normalization (same as leaderboard script) -----
    ref_dose = data_dict['dose'] * data_dict['dose_scale']
    ptv_highdose = scale_dose_Dict[patient_id]['PTV_High']['PDose']

    norm_scale = ptv_highdose / (
        np.percentile(ref_dose[data_dict[PTVHighname] > 0], 3) + 1e-8
    )
    ref_dose = ref_dose * norm_scale

    pred_path = f'{prediction_folder}/{plan_file_name}_pred.npy'
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Missing prediction file: {pred_path}")
    prediction = np.load(pred_path)

    body_mask = data_dict['Body']
    diff = ref_dose - prediction

    #################################################
    # 1) Leaderboard 5Gy MAE
    #################################################
    union_mask = ((ref_dose > 5) | (prediction > 5)) & (body_mask > 0)
    ref_mask = (ref_dose > 5) & (body_mask > 0)
    metric1 = np.sum(np.abs(diff)[union_mask]) / (np.sum(ref_mask) + 1e-8)
    Metric1_MAE_list.append(metric1)

    #################################################
    # 2) Dose Score (Body MAE)
    #################################################
    dose_score = np.mean(np.abs(diff)[body_mask > 0])
    DoseScore_list.append(dose_score)

    #################################################
    # 3) Target DVH + HI + CI (+ NEW: D1/D95/D99 diffs)
    #################################################
    ptv_mask = data_dict[PTVHighname]

    ref_target = compute_D_metrics(ref_dose, ptv_mask)
    pred_target = compute_D_metrics(prediction, ptv_mask)

    # NEW: store PTV-related diffs for table
    PTV_D1_list.append(abs(pred_target["D1"] - ref_target["D1"]))
    PTV_D95_list.append(abs(pred_target["D95"] - ref_target["D95"]))
    PTV_D99_list.append(abs(pred_target["D99"] - ref_target["D99"]))

    target_diffs = [
        abs(pred_target["D1"] - ref_target["D1"]),
        abs(pred_target["D95"] - ref_target["D95"]),
        abs(pred_target["D99"] - ref_target["D99"])
    ]

    HI_ref = compute_HI(ref_dose, ptv_mask)
    HI_pred = compute_HI(prediction, ptv_mask)
    HI_list.append(abs(HI_pred - HI_ref))

    CI_ref = compute_CI(ref_dose, ptv_mask, ptv_highdose)
    CI_pred = compute_CI(prediction, ptv_mask, ptv_highdose)
    CI_list.append(abs(CI_pred - CI_ref))

    #################################################
    # 4) OAR DVH (Dmean and D0.1cc approx)
    #################################################
    if site_list[i] == 1:
        OAR_LIST = HaN_OAR_LIST
    else:
        OAR_LIST = Lung_OAR_LIST

    oar_mean_diffs = []
    oar_d01cc_diffs = []

    for name in OAR_LIST:
        if name not in data_dict:
            continue

        mask = data_dict[name]
        if np.sum(mask) == 0:
            continue

        ref_mean = np.mean(ref_dose[mask > 0])
        pred_mean = np.mean(prediction[mask > 0])
        oar_mean_diffs.append(abs(pred_mean - ref_mean))

        ref_d01 = compute_D01cc_voxel(ref_dose, mask)
        pred_d01 = compute_D01cc_voxel(prediction, mask)
        oar_d01cc_diffs.append(abs(pred_d01 - ref_d01))

    if len(oar_mean_diffs) > 0:
        DmeanOAR_list.append(float(np.mean(oar_mean_diffs)))
        D01cc_list.append(float(np.mean(oar_d01cc_diffs)))
        # Patient-level balancing: average target part and OAR part, then average them
        oar_all = oar_mean_diffs + oar_d01cc_diffs
        dvh_score = (np.mean(target_diffs) + np.mean(oar_all)) / 2.0
    else:
        DmeanOAR_list.append(0.0)
        D01cc_list.append(0.0)
        dvh_score = float(np.mean(target_diffs))

    DVHScore_list.append(dvh_score)


#################################################
# ---------------------- Summary (Mean +- Std) ------------------------
#################################################

def print_metric(name, values):
    m, s = mean_std(values)
    print(f"{name}: {m:.4f} +- {s:.4f}")
    return m, s

print("\n===== Final Results (Mean +- Std) =====\n")

m_metric1, s_metric1 = print_metric("Leaderboard 5Gy MAE", Metric1_MAE_list)
m_dose, s_dose = print_metric("DoseScore (Body MAE)", DoseScore_list)
m_dvh, s_dvh = print_metric("DVHScore", DVHScore_list)
m_hi, s_hi = print_metric("HI", HI_list)
m_ci, s_ci = print_metric("CI", CI_list)
m_d01, s_d01 = print_metric("D0.1cc", D01cc_list)
m_dmean, s_dmean = print_metric("DmeanOAR", DmeanOAR_list)

# NEW: PTV-related metrics for Table II style
m_d1, s_d1 = print_metric("PTV D1 abs error", PTV_D1_list)
m_d95, s_d95 = print_metric("PTV D95 abs error", PTV_D95_list)
m_d99, s_d99 = print_metric("PTV D99 abs error", PTV_D99_list)


#################################################
# ---------------------- Save to CSV ------------------------
#################################################

model_name = model_name_override if model_name_override else infer_model_name(prediction_folder)

row = {
    "model": model_name,

    "leaderboard_MAE5": fmt_ms(m_metric1, s_metric1, decimal_places),
    "dose_score": fmt_ms(m_dose, s_dose, decimal_places),
    "dvh_score": fmt_ms(m_dvh, s_dvh, decimal_places),
    "HI": fmt_ms(m_hi, s_hi, decimal_places),
    "CI": fmt_ms(m_ci, s_ci, decimal_places),
    "D0.1cc": fmt_ms(m_d01, s_d01, decimal_places),
    "DmeanOAR": fmt_ms(m_dmean, s_dmean, decimal_places),

    "PTV_D1": fmt_ms(m_d1, s_d1, decimal_places),
    "PTV_D95": fmt_ms(m_d95, s_d95, decimal_places),
    "PTV_D99": fmt_ms(m_d99, s_d99, decimal_places),
}

out_df = pd.DataFrame([row])

if csv_mode == "overwrite_all":
    out_df.to_csv(csv_path, index=False)
    print(f"[CSV] Overwritten entire file: {csv_path}")

elif csv_mode == "append":
    if os.path.exists(csv_path):
        old = pd.read_csv(csv_path)
        new = pd.concat([old, out_df], ignore_index=True)
        new.to_csv(csv_path, index=False)
    else:
        out_df.to_csv(csv_path, index=False)
    print(f"[CSV] Appended new row.")

elif csv_mode == "overwrite_row":
    if os.path.exists(csv_path):
        old = pd.read_csv(csv_path)

        if "model" in old.columns and (old["model"] == model_name).any():
            # Replace only this model row
            idx = old.index[old["model"] == model_name][0]
            for col in out_df.columns:
                old.loc[idx, col] = out_df.loc[0, col]
            new = old

            new.to_csv(csv_path, index=False)
            print(f"[CSV] Overwritten row for model: {model_name}")
        else:
            new = pd.concat([old, out_df], ignore_index=True)
            new.to_csv(csv_path, index=False)
            print(f"[CSV] Model not found. Added new row.")
    else:
        out_df.to_csv(csv_path, index=False)
        print(f"[CSV] File not found. Created new CSV.")

else:
    raise ValueError("csv_mode must be one of: overwrite_all, append, overwrite_row")

