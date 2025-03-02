experiment-design
=================

.. |CDF| replace:: :abbr:`CDF (Cumulative Distribution Function)`
.. |DoE| replace:: :abbr:`DoE (Design of Experiments)`
.. |LHS| replace:: :abbr:`LHS (Latin Hypercube Sampling)`

.. image:: https://zenodo.org/badge/756928984.svg
  :target: https://doi.org/10.5281/zenodo.14635604


Easily create and extend `randomized experiment <https://en.wikipedia.org/wiki/Latin_hypercube_sampling>`_ designs with the :code:`experiment-design` library.
Currently, it supports the creation of experiment designs using

- Random sampling
- `Latin hypercube sampling <orthogonal_sampling.html#latin-hypercube-sampling>`_
- `Orthogonal sampling <orthogonal_sampling.html#orthogonal-sampling>`_

Experiment variables can be discrete and continuous as well as correlated. `experiment-design` supports all these
cases and aims to provide a space-filling designs with minimal correlation error.




.. toctree::
   :maxdepth: 2
   :caption: Documentation Contents:

   quickstart
   orthogonal_sampling
   benchmark
   api_reference
   citing
   contribution
