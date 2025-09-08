# Implementation and Evaluation of HPT on the RoboTwin Platform

This project implements the HPT (Heterogeneous Pre-trained Transformers) model and provides a full pipeline for data processing, training, and evaluation on the RoboTwin simulation platform. This repository uses Git LFS to manage the necessary pre-trained model files.

## 1. Setup

All dependencies for this project are listed in the `environment.yml` file. 

**Steps:**
1.  **Install Git LFS** (if not already installed):
    ```bash
    git lfs install
    ```

2.  **Clone this repository**:
    The `git clone` command will automatically download the large model files managed by Git LFS.
    ```bash
    git clone [https://github.com/WeijieWan77/RoboTwin-HPT.git](https://github.com/WeijieWan77/RoboTwin-HPT.git)
    cd RoboTwin-HPT
    ```

3.  **Create and activate the Conda environment using the `.yml` file**:
    ```bash
    conda env create -f environment.yml
    conda activate env_name
    ```

## 2. Data Preparation

Before training, the raw HDF5 trajectory data needs to be placed in the correct directory and pre-processed.

**Steps:**
1.  **[IMPORTANT] Place the Data**: The data processing script is configured to look for data within its own directory structure. Place the raw `.hdf5` trajectory data folders into the following specific path:
    
    `policy/HPT/HPT/data/<task_name>/<task_config>/data/`
    
    For example, for the `place_container_plate` task, the final structure should be:
    ```
    RoboTwin-HPT/
    └── policy/
        └── HPT/
            └── HPT/
                └── data/
                    └── place_container_plate/
                        └── demo_clean/
                            └── data/
                                ├── episode0.hdf5
                                └── ...
    ```
    (Note: The `policy/HPT/HPT/data/` directory is ignored by `.gitignore` and will not be uploaded to the repository).

2.  **Run the Processing Script**: Execute the data processing script. This will convert the HDF5 data into a `.pkl` cache to accelerate subsequent training runs.

    ```bash
    # Format: python <script_path> <task_name> <task_config> <num_episodes>
    # Example:
    python policy/HPT/process_data.py place_container_plate demo_clean 100
    ```

## 3. Model Training

Model training is initiated by the `train.py` script. All hyperparameters are defined in the configuration files.

**Steps:**
1.  **[IMPORTANT] Configure Task Name**:
    * Open the core configuration file located at: `policy/HPT/HPT/experiments/configs/config.yaml`.
    * **Ensure the `task_name` parameter in this file matches the name of the data folder you intend to train on.** For example, to train on the data prepared above, it must be set to:
      ```yaml
      # in config.yaml
      task_name: place_container_plate
      ```
    * This step is crucial for the script to load the correct dataset. You can also review other parameters like `episode_num`, `batch_size`, etc.

2.  **Start Training**:
    ```bash
    python policy/HPT/train.py
    ```

3.  **Training Artifacts**:
    * Model weights (`model.pth`), the configuration snapshot (`config.yaml`), and other logs will be saved to a unique, timestamped directory under `output/`. 

## 4. Model Evaluation

Model evaluation is performed using the `eval.sh` script.

**Steps:**
1.  **Configure the Evaluation Target**:
    * Open the `eval.sh` script file.
    * Find the `cfg` variable.
    * Modify its value to the path of the experiment directory from **Step 3** that you wish to evaluate. For example:
      ```bash
      # in eval.sh
      cfg="output/hpt_train/model.pth" 
      ```

2.  **Run Evaluation**:
    ```bash
    # Format: bash eval.sh <task_name> <task_config> <ckpt_setting_name> <seed> <gpu_id>
    # Example:
    bash eval.sh place_container_plate demo_clean demo_clean 0 4
    ```

3.  **Evaluation Results**:
    * Evaluation videos (`.mp4`) and result files (`_result.txt`) will be saved in the `eval_result/` directory, inside a unique sub-directory created for this evaluation run.