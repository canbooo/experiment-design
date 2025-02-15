# Benchmark LHS designers (02.2025)

Goal of the scripts in this folder is to conduct benchmark tests of the common Latin hypercube sampling libraries. The
list of libraries to test was inspired by [awesome-design-of-experiments](https://github.com/danieleongari/awesome-design-of-experiments).

## How to conduct the benchmark

First, we create an `ExternalDesignerAdapter` for each tested library. Following, we use the `generate_experiment_design.py`
script to generate designs of experiments (DoE). We only create uniform DoEs with libraries that do not natively support
non-uniform distributions to avoid unnecessary computations. This does not change the results for those libraries as
mapping the DoEs to different non-uniform spaces would result in the same DoE as using the `ExternalDesignerAdapter`
methods due to the fixed random seeds. For mapping uniform probabilities to non-uniform design space, we use
`generate_non_native_orthogonal_design.py`. Following, we run `evaluate_doe.py` to compute various DoE metrics that are
implemented in `metrics.py`. Finally, we use `result_plots_tables.py` to generate the plots that we insert into our
documentation.
