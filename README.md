# Weighted Euclidean Distance Matrices over Mixed Inputs for Gaussian Process Models.
**WEGP** (Weighted Euclidean Gaussian Process) is designed to handle mixed input types by employing a weighted Euclidean distance matrix that captures differences between continuous and categorical variables more effectively. This respository contains code to run the experiments in the paper [Weighted Euclidean Distance Matrices over Mixed Inputs for
Gaussian Process Models](https://openreview.net/forum?id=BuhgHqJQV7). 

For reproducing the experiments, refer to the each subdirectory in the `tests/` folder.

## Installation

```bash
pip install -r requirements.txt
```

### Training the WEGP Model

To train the WEGP model, run the following command:

```bash
python ./tests/functions/run_script.py --save_dir results --which_func func2C --train_factor 1 --n_jobs 1 --n_repeats 1
```
**Parameter Descriptions:**
- `--save_dir`: Directory to save the experiment results.
- `--train_factor`: A factor to adjust the scale of training.
- `--which_func`: Selects the test function (e.g., `func2C`).
- `--n_jobs`: Number of parallel jobs.
- `--n_repeats`: Number of times to repeat the experiment.
- `--budget`: Budget parameter controlling the number of function evaluations.



### Running the WEBO Algorithm

Execute the following command to run the WEBO algorithm for optimization experiments. You can adjust the experimental settings using the command-line parameters:

```bash
python ./tests/functions/optimization.py --save_dir results --train_factor 1 --which_func func2C --n_jobs 1 --n_repeats 1 --budget 100
```



Parameters function similarly to those in the WEBO algorithm command, specifying experiment settings and configurations.

## Citation
```bibtex
@inproceedings{
  pu2025weighted,
  title={Weighted Euclidean Distance Matrices over Mixed Continuous and Categorical Inputs for Gaussian Process Models},
  author={Mingyu Pu and Wang Songhao and Haowei Wang and Szu Hui Ng},
  booktitle={The 28th International Conference on Artificial Intelligence and Statistics},
  year={2025},
  url={https://openreview.net/forum?id=BuhgHqJQV7}
}
```


Below is a comprehensive README example in English that you can use as a template for your repository:

---

# Weighted Euclidean Distance Matrices over Mixed Inputs for Gaussian Process Models

This repository contains the code for experiments presented in the paper “[Weighted Euclidean Distance Matrices over Mixed Inputs for Gaussian Process Models](https://openreview.net/forum?id=BuhgHqJQV7)”. The model addresses problems with mixed inputs (continuous and categorical) within Gaussian Process models.

## Project Overview


