# LiFe-net: Data-driven Modelling of Time-dependent Temperatures and Charging Statistics Of Tesla’s LiFePo4 EV Battery


## How to run
```python LiFe-net_baseline.py```

```python LiFe-net_regularised.py```

```python LiFe-net_t_stability.py```

## How to load pre-trained model

```
model.load_state_dict(torch.load(PATH))
```

## How to do hyperparameter optimization

See documentations of Weights and Biases library:
https://docs.wandb.ai/guides/sweeps

Example:

```python -m wandb sweep sweep-alpha.yaml```

``` python -m wandb agent 'name/of/the/agent' ```
