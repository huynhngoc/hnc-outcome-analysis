# Head and Neck cancer treatment outcome analysis

All experiments were run by submitting to the slurm scheduling system on Orion (The High Computing Clusters of NMBU), which requires an slurm script (`.sh`), a container (we used Singularity `*.sif`) and the code file (python `*.py`).


## Singularity definition file
`Singularity.ray`

Manually build the container using
```
singularity build --fakeroot Singularity.ray deoxys.sif
```

## 3D EfficientNet
Check the `new_layer.py` and `customize_obj.py` files, together with the `architecture` & `scripts` folders.

## Tradition ML experiments
See `outcome_traditional.py` and `outcome_traditional_radiomics.py`

## Neural net configuration
See `config/clinical*`, `config/radiomics*`, and `config/tabular*` folders

## CNN configuration
See `config/outcome_img*` folders


## Bootstrap sampling results
See `outcome_plot.py`

## Interpretability
To generate vargrad, see `interpretability.py` & `interpretability.sh`
To analyze results, see `interpret_analyze.py` & `interpret_analyze.sh`

## Figures generation
See `outcome_vargrad_plot.py`
