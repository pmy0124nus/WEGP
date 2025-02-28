import os
import time
import torch
import random
import argparse
import numpy as np
from joblib import dump, Parallel, delayed, load
from scipy.stats import norm
import configs
import functions
import configs
from wegp_bayes.models import WEGP #TODO
from numpyro.diagnostics import summary
from wegp_bayes.utils.metrics import rrmse,mean_interval_score,coverage
from wegp_bayes.utils.metrics import gaussian_mean_confidence_interval

import itertools
from wegp_bayes.optim import fit_model_scipy
from wegp_bayes.optim import run_hmc_numpyro_wegp
# for MCMC
import jax
jax.config.update("jax_enable_x64", True)

parser = argparse.ArgumentParser('Engg functions MAP vs fully Bayesian')
parser.add_argument('--save_dir', type=str, required=True)
parser.add_argument('--which_func', type=str, required=True)
parser.add_argument('--train_factor', type=int, required=True)
parser.add_argument('--n_jobs', type=int, required=True)
parser.add_argument('--n_repeats', type=int, default=25)
parser.add_argument('--maxfun', type=int, default=500)
parser.add_argument('--noise', type=bool, default=False)
parser.add_argument('--model', type=str, default='WEGP_0919', choices=['WEGP_0919','WEGP','WEGP_0828', 'LVGP']) #TODO
parser.add_argument('--budget', type=int, default=200)
parser.add_argument('--num_permutations',type=int,default=0) #TODO

args = parser.parse_args()
func = args.which_func
save_dir = os.path.join(
    args.save_dir,
    '%s/train_factor_%d' % (args.which_func, args.train_factor),
)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

config_fun = getattr(configs, func)()
obj = getattr(functions, func)
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

def generate_latents(num_levels, num_permutations):
    all_permutations = list(itertools.permutations(range(num_levels)))

    def sample_permutations(all_permutations, num_permutations):
        selected_indices = np.random.choice(len(all_permutations), num_permutations, replace=False)
        selected_permutations = [all_permutations[i] for i in selected_indices]
        return selected_permutations

    def is_full_rank(permutations, num_levels, num_permutations):
        num_distances = num_levels * (num_levels - 1) // 2
        distance_matrix = torch.zeros((num_distances, num_permutations))
        
        for i, perm in enumerate(permutations):
            distances = [abs(perm[j] - perm[k]) for j in range(num_levels) for k in range(j + 1, num_levels)]
            distance_matrix[:, i] = torch.tensor(distances)
        return torch.linalg.matrix_rank(distance_matrix) >= min(distance_matrix.size())

    while True:
        permutations = sample_permutations(all_permutations, num_permutations)
        if is_full_rank(permutations, num_levels, num_permutations):
            perm_tensor = torch.tensor(permutations).T.float()
            return perm_tensor

test_x = torch.from_numpy(config_fun.random_sample(np.random.RandomState(456),1000))
test_y = [None]*test_x.shape[0]
for i,x in enumerate(test_x):
    test_y[i] = obj(config_fun.get_dict_from_array(x.numpy())) #config.get_dict_from_array(test_x[0].numpy()):  {'r': 12512.919876919379, 'T_u': 106776.98444579469, 'H_u': 1099.5286375868927, 'T_l': 100.20007435323, 'L': 1617.2506666329225, 'K_w': 11880.333581127572, 'r_w': 3, 'H_l': 0}

# create tensor objects

test_y = torch.tensor(test_y).to(test_x)

# 将标准化样本转换到原始的现实范围
mapped_test_x_list = []
for x in test_x:
    # 调用 get_dict_from_array 得到每个样本在原始尺度上的字典表示
    x_dict = config_fun.get_dict_from_array(x)
    # 根据变量的顺序构建数组（假设 config_fun.get_variables() 返回的顺序即为你希望保存的顺序）
    x_mapped = np.array([x_dict[var.name] for var in config_fun.get_variables()])
    
    mapped_test_x_list.append(x_mapped)
    
# 转换为 torch.Tensor（这里建议明确指定数据类型）
mapped_test_x = torch.tensor(np.array(mapped_test_x_list), dtype=torch.float32)

# 保存转换后的训练数据：映射到现实范围的 x 和对应的 y
torch.save(mapped_test_x, os.path.join(save_dir, 'test_x.pt'))
torch.save(test_y, os.path.join(save_dir, 'test_y.pt'))

def main_script(seed):
    save_dir_seed = os.path.join(save_dir, 'seed_%d' % seed)
    if not os.path.exists(save_dir_seed):
        os.makedirs(save_dir_seed)

    num_levels_per_var = list(config_fun.num_levels.values())
    n_train = args.train_factor
    for n in num_levels_per_var:
        n_train *= n

    rng = np.random.RandomState(seed)
    train_x = torch.from_numpy(config_fun.latinhypercube_sample(rng, n_train))
    train_y = [obj(config_fun.get_dict_from_array(x.numpy())) for x in train_x]
    train_y = torch.tensor(train_y).to(train_x)

    # 保存训练数据
    # 生成标准化的训练样本（在 [0,1] 范围内）
    

    # 将标准化样本转换到原始的现实范围
    mapped_train_x_list = []
    for x in train_x:
        # 调用 get_dict_from_array 得到每个样本在原始尺度上的字典表示
        x_dict = config_fun.get_dict_from_array(x)
        # 根据变量的顺序构建数组（假设 config_fun.get_variables() 返回的顺序即为你希望保存的顺序）
        x_mapped = np.array([x_dict[var.name] for var in config_fun.get_variables()])
        
        mapped_train_x_list.append(x_mapped)
        
    # 转换为 torch.Tensor（这里建议明确指定数据类型）
    mapped_train_x = torch.tensor(np.array(mapped_train_x_list), dtype=torch.float32)

    # 保存转换后的训练数据：映射到现实范围的 x 和对应的 y
    torch.save(mapped_train_x, os.path.join(save_dir_seed, 'train_x.pt'))
    torch.save(train_y, os.path.join(save_dir_seed, 'train_y.pt'))

    latents_list = []
   
    set_seed(seed)

    def default_permutation_num(input_list):
        return [(n * (n - 1)) // 2 for n in input_list]
    
    if args.num_permutations==0:
        num_permutations = default_permutation_num(num_levels_per_var)
    else:
        num_permutations = [args.num_permutations]*len(num_levels_per_var)
        print(f'num_permutations: {num_permutations}')

    for num_levels, num_perms in zip(num_levels_per_var, num_permutations):
        latents = generate_latents(
            num_levels=num_levels,
            num_permutations=num_perms
            )  
        latents_list.append(latents)
    model = WEGP(
        train_x=train_x,
        train_y=train_y,
        quant_correlation_class='Matern32Kernel',
        qual_index=config_fun.qual_index,
        quant_index=config_fun.quant_index,
        num_levels_per_var=num_levels_per_var,
        num_permutations = num_permutations,
        latents_list=latents_list,
        noise=torch.tensor(0.25).double() if args.noise else None,
        fix_noise=args.noise)

    jax.config.update("jax_enable_x64", True) 
    start_time = time.time()
    mcmc_runs = run_hmc_numpyro_wegp(
        model,
        latents_list,
        num_samples=1500,warmup_steps=1500,
        max_tree_depth=7,
        disable_progbar=True,
        num_chains=1,
        num_model_samples=100,
        seed=seed
    )
    fit_time = time.time() - start_time
    print("fit_time: ", fit_time)
    import pickle
    diagnostics = summary(mcmc_runs.get_samples(),group_by_chain=False)
    with open(os.path.join(save_dir_seed,'mcmc_diagnostics.pkl'),'wb') as f:
        pickle.dump(diagnostics,f)

        # predictions
        with torch.no_grad():
            means,stds = model.predict(test_x,return_std=True)
        
        lq,uq = gaussian_mean_confidence_interval(means,stds)
        
        stats = {
            'rrmse':rrmse(test_y,means.mean(axis=0)).item(),
            'mis':mean_interval_score(test_y,lq,uq,0.05).item(),
            'coverage':coverage(test_y,lq,uq).item(),
            'training_time':fit_time
        }
    
        # 保存模型和结果
        torch.save(model.state_dict(), os.path.join(save_dir_seed, f'{args.model}_{args.estimation}_buget_{args.budget}_{args.num_permutations}_state.pth'))
        dump(stats, os.path.join(save_dir_seed, f'stats_{args.model}_buget_{args.budget}_optimization_num_permutation_{args.num_permutations}.pkl'))


seeds = np.linspace(100,1000,args.n_repeats).astype(int)

Parallel(n_jobs=args.n_jobs,verbose=0)(
    delayed(main_script)(seed) for seed in seeds
)
