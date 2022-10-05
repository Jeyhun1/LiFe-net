# LiFe-net: Data-driven Modelling of Time-dependent Temperatures and Charging Statistics Of Tesla’s LiFePo4 EV Battery

## Requirements

### Libaries
```
cuda 11.1
python/3.7.4
```

### Python requirements
```
torch>=1.8.1
numpy>=1.19.5
matplotlib>=3.1.1
```


## How to run
```python LiFe-net_baseline.py```

```python LiFe-net_regularised.py```

```python LiFe-net_t_stability.py```

## How to load pre-trained model
### Load models from 'models' folder as such:
```
model.load_state_dict(torch.load(PATH))
```
## Plots included in the paper
See ```evaluation_plots.ipynb``` Jupyter Notebook

## How to do hyperparameter optimization

See documentations of Weights and Biases library:
https://docs.wandb.ai/guides/sweeps

Example:

```python -m wandb sweep sweep-alpha.yaml```

``` python -m wandb agent 'name/of/the/agent' ```
