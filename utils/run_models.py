import os
import subprocess

def run_surrogate(test_file='tmp/species_reactivity_dataset.csv'):

        path = "surrogate_model/predict.py"
        test_path = f"{test_file}"
        chk_path = "surrogate_model/qmdesc_wrap/model.pt"
        preds_path = "tmp/preds_surrogate.pkl"
        inputs = f"--test_path {test_path} --checkpoint_path {chk_path} --preds_path {preds_path}"

        with open('out_file', 'a') as out:
            subprocess.run(f"python {path} {inputs}", shell=True, stdout=out, stderr=out)

        return None


def run_reactivity(trained_dir = 'reactivity_model/results/final_model_4/', target_column='DG_TS_tunn', ensemble_size=4, input_file='input_ffnn.csv'):

    path = f"reactivity_model/predict.py"
    pred_file = f"tmp/{input_file}"
    save_dir = "tmp/"
    inputs = f"--pred_file {pred_file} --trained_dir {trained_dir} --save_dir {save_dir} --ensemble_size {ensemble_size} --target_column {target_column}"

    with open('out_file', 'a') as out:
        subprocess.run(f"python {path} {inputs}", shell=True, stdout=out, stderr=out)

    return None


def run_cv(data_path = None, 
           target_column = 'DG_TS_tunn', 
           save_dir = 'tmp/', 
           k_fold = 10,
           ensemble_size = 4, 
           sample = None, 
           transfer_learning = False,
           trained_dir = 'reactivity_model/results/final_model_4/',
           random_state = 2,
           test_set = None,
           train_valid_set = None, 
           batch_size = 64):
    
    path = f"reactivity_model/cross_val.py"
    inputs = f" --k_fold {k_fold} --hidden-size 230 --target_column {target_column} --batch-size {batch_size} \
    --learning_rate 0.0277 --lr_ratio 0.95 --random_state {random_state} --ensemble_size {ensemble_size}  --save_dir {save_dir}"
    if data_path:
         inputs += f" --data_path {data_path}"
    if sample:
         inputs += f" --sample {sample}"
    if transfer_learning:
         inputs += f" --transfer_learning --trained_dir {trained_dir}"
    if test_set and train_valid_set:
         inputs += f" --test_set_path {test_set}  --train_valid_set_path {train_valid_set}"
         

    with open('out_file', 'a') as out:
        subprocess.run(f"python {path} {inputs}", shell=True, stdout=out, stderr=out)
    
    return None


def run_train(save_dir, 
              data_path = 'tmp/input_ffnn.pkl', 
              trained_dir = 'reactivity_model/results/final_model_4/', 
              transfer_learning = True, 
              target_column='DG_TS_tunn',
              ensemble_size=4,
              batch_size=64):
     
    path = f"reactivity_model/train.py"
    inputs = f" --data_path {data_path} --save_dir {save_dir} --ensemble_size {ensemble_size} --target_column {target_column} --batch-size {batch_size}"
    if transfer_learning:
         inputs += f" --trained_dir {trained_dir} --transfer_learning"

    with open('out_file', 'a') as out:
        subprocess.run(f"python {path} {inputs}", shell=True, stdout=out, stderr=out)
    
    return None


    
