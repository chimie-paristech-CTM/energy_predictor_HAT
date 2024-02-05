import os
import subprocess

def run_surrogate():

        path = f"surrogate_model/predict.py"
        test_path = "tmp/species_reactivity_dataset.csv"
        chk_path = "surrogate_model/qmdesc_wrap/model.pt"
        preds_path = "tmp/preds_surrogate.pkl"
        inputs = f"--test_path {test_path} --checkpoint_path {chk_path} --preds_path {preds_path}"

        with open('out_file', 'w') as out:
            subprocess.run(f"python {path} {inputs}", shell=True, stdout=out, stderr=out)

        return None


def run_reactivity():

    path = f"reactivity_model/predict.py"
    pred_file = "tmp/input_ffnn.csv"
    trained_dir = "reactivity_model/results/final_model_4/"
    save_dir = "tmp/"
    inputs = f"--pred_file {pred_file} --trained_dir {trained_dir} --save_dir {save_dir} --ensemble_size 4"

    with open('out_file', 'w') as out:
        subprocess.run(f"python {path} {inputs}", shell=True, stdout=out, stderr=out)

    return None


def run_cv(data_path, target_column, save_dir):
    
    path = f"reactivity_model/cross_val.py"
    inputs = f" --data_path {data_path} --k_fold 10 --hidden-size 230 --target_column {target_column} \
    --learning_rate 0.0277 --lr_ratio 0.95 --random_state 2 --ensemble_size 4  --save_dir {save_dir}"

    with open('out_file', 'w') as out:
        subprocess.run(f"python {path} {inputs}", shell=True, stdout=out, stderr=out)
    
    return None

    
