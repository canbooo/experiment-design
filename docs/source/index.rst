experiment-design
=================

.. |CDF| replace:: :abbr:`CDF (Cumulative Distribution Function)`
.. |DoE| replace:: :abbr:`DoE (Design of Experiments)`
.. |LHS| replace:: :abbr:`LHS (Latin Hypercube Sampling)`

experiment-design allows anyone to easily create and extend randomized experiment designs. Currently, it supports
creating designs using

- Random sampling
- `Latin hypercube sampling <orthogonal_sampling.html#latin-hypercube-sampling>`_
- `Orthogonal sampling <orthogonal_sampling.html#orthogonal-sampling>`_

Experiment variables can be discrete and continuous as well as correlated. `experiment-design` can handle all of these
cases and seeks to provide a space-filling design of experiments with a small correlation error.





.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quickstart
   orthogonal_sampling
   api_reference
