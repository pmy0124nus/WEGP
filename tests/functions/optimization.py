import os
import torch
import random
import argparse
import numpy as np
from joblib import dump, Parallel, delayed
from scipy.stats import norm
from tqdm import tqdm
import configs
import functions
import configs
from wegp_bayes.models import WEGP
import itertools
from wegp_bayes.optim import run_hmc_numpyro_wegp
import jax
jax.config.update("jax_enable_x64", True)

parser = argparse.ArgumentParser('Engg functions fully Bayesian')
parser.add_argument('--save_dir', type=str, required=True)
parser.add_argument('--which_func', type=str, required=True)
parser.add_argument('--train_factor', type=int, required=True)
parser.add_argument('--n_jobs', type=int, required=True)
parser.add_argument('--n_repeats', type=int, default=25)
parser.add_argument('--maxfun', type=int, default=500)
parser.add_argument('--noise', type=bool, default=False)
parser.add_argument('--budget', type=int, default=200)
parser.add_argument('--num_permutations',type=int,default=0)

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

class UCB:
    def __init__(self, model, kappa=2.576):
        self.model = model
        self.kappa = kappa

    def evaluate(self, x):
        x = torch.tensor(x, dtype=torch.float64)
        self.model.eval()
        with torch.no_grad():
            mu, sigma = self.model.predict(x, return_std=True)
        return mu.numpy() + self.kappa * sigma.numpy()

class EI:
    def __init__(self, model, best_f):
        self.model = model
        self.best_f = best_f

    def evaluate(self, x):
        self.model.eval()
        with torch.no_grad():
            mu, sigma = self.model.predict(x, return_std=True)
        Z = (mu.numpy() - self.best_f) / sigma.numpy()
        return (mu.numpy() - self.best_f) * norm.cdf(Z) + sigma.numpy() * norm.pdf(Z)
class EI_NUTS:
    def __init__(self, model, best_f):
        """
        Initialize the EI object.

        Args:
        - model: The GP model that provides the predictions (mean and std).
        - best_f: The current best (minimum) value observed in the optimization process.
        """
        self.model = model
        self.best_f = best_f

    def evaluate(self, x, num_samples=100):
        """
        Evaluate the Expected Improvement (EI) at a given point `x`.
        
        Args:
        - x: Candidate points for evaluation (could be a batch of points).
        - num_samples: The number of posterior samples used for the EI computation.

        Returns:
        - The expected improvement for the given points `x`.
        """
        x = torch.tensor(x, dtype=torch.float64)
        self.model.eval() 

        ei_per_sample = []
        with torch.no_grad():
            mu, sigma = self.model.predict(x, return_std=True)
            for i in range(num_samples):
                mu_i = mu[i].numpy()   
                sigma_i = sigma[i].numpy()  
                
                Z = (mu_i - self.best_f) / sigma_i
                ei = (mu_i - self.best_f) * norm.cdf(Z) + sigma_i * norm.pdf(Z)
                ei_per_sample.append(ei)

        avg_ei = np.mean(ei_per_sample, axis=0)
        return avg_ei

def get_acq_values(acq_object, design_array):
    num_chunks = 10  
    x_chunks = np.array_split(design_array, num_chunks)
    f_samples_list = []
    for x_chunk in x_chunks:
        f_samples_list.append(acq_object.evaluate(x_chunk))
    return np.concatenate(f_samples_list).ravel()

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

    torch.save(train_x, os.path.join(save_dir_seed, 'train_x.pt'))
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
        # exit()

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
    run_hmc_numpyro_wegp(
        model,
        latents_list,
        num_samples=1500,warmup_steps=1500,
        max_tree_depth=7,
        disable_progbar=True,
        num_chains=1,
        num_model_samples=100,
        seed=seed
    )

    best_y_list = []
    for iteration in tqdm(range(args.budget)): 
        design_data = torch.from_numpy(config_fun.random_sample(np.random.RandomState(iteration),500)) 
        best_f = train_y.max().item()
        best_y_list.append(best_f)
        acq_object = EI_NUTS(model, best_f)
        acq_values = get_acq_values(acq_object, design_data)
        print(f"Best f: {best_f}")

        next_index = np.argmax(acq_values)
        next_x = torch.tensor(design_data[next_index], dtype=torch.float64).unsqueeze(0)
        next_y = obj(config_fun.get_dict_from_array(next_x.numpy()[0]))
        train_x = torch.cat([train_x, next_x], dim=0)
        train_y = torch.cat([train_y, torch.tensor([next_y], dtype=torch.float64)], dim=0)

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
        run_hmc_numpyro_wegp(
            model,
            latents_list,
            num_samples=1500,warmup_steps=1500,
            max_tree_depth=7,
            disable_progbar=True,
            num_chains=1,
            num_model_samples=100,
            seed=seed
        )
    best_f = train_y.max().item()
    best_y_list.append(best_f)
    stats = {
        'best_y_list': np.array(best_y_list),
    }
    torch.save(model.state_dict(), os.path.join(save_dir_seed, f'buget_{args.budget}_{args.num_permutations}_state.pth'))
    dump(stats, os.path.join(save_dir_seed, f'buget_{args.budget}_optimization_num_permutation_{args.num_permutations}.pkl'))


seeds = np.linspace(100,1000,args.n_repeats).astype(int)

Parallel(n_jobs=args.n_jobs,verbose=0)(
    delayed(main_script)(seed) for seed in seeds
)
