# QASER

Code for paper [QASER: Breaking the Depth vs. Accuracy Trade-Off for Quantum Architecture Search](https://arxiv.org/abs/2511.16272)

-> **QASER** take into account seemingly contradictory optimization goals by introducing a reward that enables the compilation of circuits with lower depth and higher accuracy, significantly outperforming state-of-the-art techniques.

## To run the code with exponential reward use the following:

```
python main.py --seed 123 --config lbmt_cobyla_8qH2O_step_250_F0_energy_untweaked --experiment_name "finalize/"
```
