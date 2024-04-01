## 1. To reproduce the simulation results in the paper:

### 1.1. Generate the random graphs:

Run the block of line 507-516 in `main.py`. The graphs will be saved in the folder `./graphs/`.

### 1.2. Simulate the data:

Run the block of line 528-560 in `main.py`. The data are simulated according to the generated graphs in step 1, with different SEM models and different dropout mechanisms. Generated data will be saved in the folder `./data/`.

### 1.3. Run the experiments:

Below is an examplar command to run:

```bash
python main.py --dagid 0
               --gtype ER
               --funcform linear
               --exonoise gaussian 
               --nonnegative raw 
               --dropouts latent.CAR.truncated.logistic
```

where different configurations to run the experiments can be specified by the arguments. The results will be saved in corresponding subfolders in `./data/`.


## 2. To run PC or GES with zero-deletion on your own scRNA-seq data:

Simply initialize the corresponding CI tester and then pass to PC or GES. Here is an example:

```python
tester = cit.CIT(
    your_own_data, # np.ndarray, in shape (n_samples, n_features)
    method="zerodel_fisherz" # or "zerodel_kci"
)
cg = pc(
    your_own_data,
    alpha=0.05, 
    indep_test=tester
)
```

For more details on visualizing / benchmarking the results, please refer to the `run_pc` (line 441) and `run_ges` (line 471) functions in `main.py`.

For the details of reframing the GES by CI tests without any score functions (and for sample size correction), please refer to e.g., line 521-556 block in `./csl/search/ScoreBased/GES.py` (implemented based on [this repo](https://github.com/juangamella/ges)).

## 3. To incorporate zero-deletion into other GRNI methods:

Please refer to e.g., line 177-228 block and line 255-294 block in `./csl/utils/cit.py` for the details in zero-deleted CI tests (implemented based on [this repo](https://github.com/py-why/causal-learn)).

Note that given our causal dropout models, the current codes only delete data samples with zeroes in conditioning variables. You may also try to delete data samples with zeroes in all involved variables (i.e., X, Y, and the conditioning variables), which is still correct (according to Def 2 and Thm 2 in the paper), though it may be less powerful with fewer sample size.