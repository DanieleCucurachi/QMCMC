# QMCMC

## Quantum-enhanced Monte Carlo markov chain optimization

The combination of classical Monte Carlo Markov chains (MCMC) methods with quantum computers showed potential for achieving quantum advantage in sampling from the Boltzmann probability distribution, a computationally hard task arising in many diverse fields [[1]](https://doi.org/10.1038/s41586-023-06095-4). Quantum-enhanced proposal distributions, defined by parameterized unitaries, can outperform classical strategies in proposing effective moves in MCMC. However, it is crucial to carefully tune the values of the parameters defining these distributions, as they determine the resulting advantage over the classical counterpart. A general optimization method becomes essential when considering problems where is not possible to identify a reasonable parameter set a priori. This could happen when adopting complicated proposal strategies depending on a large number of parameters, or simply when no prior or relevant information is available.

This repository contains the python implementation of a general optimization subroutine for parametrized proposal distributions.

See ```tutorial.ipynb``` for a usage example.

## References

[[1]](https://doi.org/10.1038/s41586-023-06095-4) Layden, D., Mazzola, G., Mishmash, R.V. et al. Quantum-enhanced Markov chain Monte Carlo. Nature 619, 282–287 (2023). https://doi.org/10.1038/s41586-023-06095-4

## License

The package is licensed under  ```MIT License```

