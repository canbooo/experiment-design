.. |DoE| replace:: :abbr:`DoE (Design of Experiments)`
.. |LHS| replace:: :abbr:`LHS (Latin Hypercube Sampling)`

Benchmarking some Latin hypercube sampling libraries
''''''''''''''''''''''''''''''''''''''''''''''''''''''

As of February 2025, the :code:`experiment-design` library **outperforms all other tested |LHS| implementations**,
achieving the **lowest correlation error** and **best space-filling properties**.

Key advantages of :code:`experiment-design`:

* **Lowest correlation error** across all dimensions and sample sizes
* **Best space-filling properties**, with higher minimum pairwise distance
* **Only library** with native support for correlated variables, non-uniform distributions, and |LHS| extension

Results
-------

Correlation error
=================

The plots below show mean and maximum correlation error results, grouped by dimension and sample size.
Since these are error metrics, lower values indicate better performance. Lines represent the average values of the metrics, whereas the areas
represent the 95\% confidence intervals.


.. image:: images/benchmark/max_correlation_error-dimension-all_distributions.png
    :align: left
    :width: 320px
    :alt: Maximum correlation error over dimensions

.. image:: images/benchmark/max_correlation_error-sample-all_distributions.png
    :align: right
    :width: 320px
    :alt: Maximum correlation error over samples

.. image:: images/benchmark/mean_correlation_error-dimension-all_distributions.png
    :align: left
    :width: 320px
    :alt: Mean correlation error over dimensions

.. image:: images/benchmark/mean_correlation_error-sample-all_distributions.png
    :align: right
    :width: 320px
    :alt: Mean correlation error over samples

Even when restricting the analysis to uniform distributions as given below, :code:`experiment-design` consistently achieves
lower correlation error, particularly in lower-dimensional settings.

.. image:: images/benchmark/max_correlation_error-dimension-uniform_distribution.png
    :align: left
    :width: 320px
    :alt: Maximum correlation error over dimensions for uniform |DoE|

.. image:: images/benchmark/max_correlation_error-sample-uniform_distribution.png
    :align: right
    :width: 320px
    :alt: Maximum correlation error over samples for uniform |DoE|

.. image:: images/benchmark/mean_correlation_error-dimension-uniform_distribution.png
    :align: left
    :width: 320px
    :alt: Mean correlation error over dimensions for uniform |DoE|

.. image:: images/benchmark/mean_correlation_error-sample-uniform_distribution.png
    :align: right
    :width: 320px
    :alt: Mean correlation error over samples for uniform |DoE|


Pairwise distance
=================

These metrics assess the space-filling properties of the |DoE| .

- Higher minimum pairwise distance is better.
- Lower inverse average distance is better.

Again, lines represent the average values of the metrics, whereas the areas represent the 95\% confidence intervals.

.. image:: images/benchmark/min_pairwise_distance-dimension-all_distributions.png
    :align: left
    :width: 320px
    :alt: Minimum pairwise distance over dimensions

.. image:: images/benchmark/min_pairwise_distance-sample-all_distributions.png
    :align: right
    :width: 320px
    :alt: Minimum pairwise distance over samples

.. image:: images/benchmark/inv_avg_distance-dimension-all_distributions.png
    :align: left
    :width: 320px
    :alt: Inverse average distance over dimensions

.. image:: images/benchmark/inv_avg_distance-sample-all_distributions.png
    :align: right
    :width: 320px
    :alt: Inverse average distance over samples


Even when considering only uniform distributions as given below, :code:`experiment-design` maintains a significant advantage.


.. image:: images/benchmark/min_pairwise_distance-dimension-uniform_distribution.png
    :align: left
    :width: 320px
    :alt: Minimum pairwise distance over dimensions for uniform |DoE|

.. image:: images/benchmark/min_pairwise_distance-sample-uniform_distribution.png
    :align: right
    :width: 320px
    :alt: Minimum pairwise distance over samples for uniform |DoE|

.. image:: images/benchmark/inv_avg_distance-dimension-uniform_distribution.png
    :align: left
    :width: 320px
    :alt: Inverse average distance over dimensions for uniform |DoE|

.. image:: images/benchmark/inv_avg_distance-sample-uniform_distribution.png
    :align: right
    :width: 320px
    :alt: Inverse average distance over samples for uniform |DoE|


Tested libraries
----------------

A non-exhaustive list of further |LHS| libraries available in python is given in `this repository <https://github.com/danieleongari/awesome-design-of-experiments>`_,
which inspired this benchmark.

- `pyDOE <https://github.com/danieleongari/awesome-design-of-experiments>`_, `pyDOE2 <https://github.com/clicumu/pyDOE2>`_,
  `pyDOE3 <https://pydoe3.readthedocs.io/en/latest/>`_: These libraries are among the first search results when looking
  for |DoE| tools, as the original pyDOE is one of the oldest Python libraries supporting |DoE| generation. They primarily
  focus on factorial designs and their derivatives but also include |LHS| functionality.

  - pyDOE2 is a direct fork of pyDOE, fixing some bugs and introducing generalized subset design. However, no changes were
    made to the |LHS| code.
  - These libraries support two |LHS| objectives: :code:`maximin` which maximizes the minimum distance and :code:`correlation` which minimizes
    the maximum correlation coefficient. Since a choice is required when using the :code:`lhs` function, pyDOE was tested
    with :code:`maximin`, while pyDOE2 was tested with :code:`correlation`.
  - pyDOE3 introduces an additional |LHS| objective, :code:`lhsmu`, which was used in this benchmark. See the linked
    documentation for further details.
- `doepy <https://doepy.readthedocs.io/en/latest/>`_: This library looked promising, especially with a function called
  :code:`space_filling_lhs`. However, using this function currently results in an unresolved reference error. Therefore,
  the standard :code:`lhs` function was used instead.

  - doepy appears to be the only library natively supporting orthogonal sampling (i.e., |DoE| with non-uniform marginal
    distributions). However, passing distributions to its function does not seem to have any effect.
  - As a result, only uniform |DoE| were generated and mapped to non-uniform distributions via `inverse transform sampling <https://en.wikipedia.org/wiki/Inverse_transform_sampling>`_,
    just as with the other libraries.
  - If these issues are resolved in future updates, the benchmark results can be recomputed.

- `diversipy <https://diversipy.readthedocs.io/en/latest/index.html>`_: This library is particularly useful due to its
  extensive |DoE| evaluation metrics in the indicator module. One of these,`average_inverse_dist <https://diversipy.readthedocs.io/en/latest/indicator.html>`_,
  was compelling enough to be included as an additional benchmark metric.

  - The function :code:`cube.improved_latin_design` was used to generate |LHS| samples.

- `pyLHD <https://github.com/toledo60/pyLHD>`_: As one of the older |DoE| libraries, pyLHD implements various |LHS| methods.
  However, most of them impose constraints on the number of dimensions and samples.

  - The :code:`maximinLHD` function was used in this benchmark, as it provided the most flexibility in terms of sample size.
  - The ability to generate flexible sample sizes is particularly important in modern computing, as it allows for efficient
    parallelization across arbitrary numbers of CPU cores or worker nodes.

Methods and metrics
-------------------

For each algorithm, 64 |DoE| were generated for every combination of dimension, sample size, and distribution.
Four probability distributions from the `scipy.stats` module were tested:

.. code:: python

    stats.uniform(loc=0, scale=1),  # Uniform distribution [0, 1]
    stats.norm(loc=0.5, scale=1 / 6.180464612335626),  # Normal distribution ~ [0, 1] (95% of values)
    stats.lognorm(0.448605225, scale=0.25),  # Log-normal distribution ~ [0, 1] (95% of values)
    stats.gumbel_r(loc=-2.81, scale=1.13),  # Gumbel distribution ~ [-5, 5] (95% of values)

The table below shows the number of dimensions and corresponding sample sizes used for each distribution:

.. list-table::
    :header-rows: 1
    :align: center

    * - Dimensions
      - Samples
    * - 2, 3, 4, 5
      - 32, 64, 96, 128
    * - 10, 15, 20, 25
      - 64, 96, 128, 256
    * - 50, 75, 100
      - 128, 256, 512

In total, 41 different (dimension, sample size) combinations were tested, across 4 distributions and 64 trials each, yielding
:math:`41 \times 4 \times 64 = 10,496` results per algorithm. Powers of two were chosen for sample sizes to align with
common parallel computing setups, where computations are distributed across a power-of-two number of CPU cores or worker nodes.

Evaluation metrics
==================
The following metrics were used:

1. Correlation Error

   - Maximum and mean correlation error (lower values are better).
   - Only non-correlated variables were considered to ensure fair comparisons.

2. Space-Filling Properties

   - Minimum pairwise distance (higher is better).
   - Inverse average distance (lower is better).

For libraries lacking native support for non-uniform distributions, uniform |DoE| were generated first and then transformed
using `inverse transform sampling <https://en.wikipedia.org/wiki/Inverse_transform_sampling>`_.




Conclusion
-----------

:code:`experiment-design` consistently produces the highest-quality |LHS| and orthogonal sampling designs.
This benchmark demonstrates that even the closest competing library performs significantly worse in at least 95\% of tested
cases. The full benchmark code is available in the `benchmark-2025-02 branch <https://github.com/canbooo/experiment-design/tree/benchmark-2025-02>`_,
and all generated |DoE| can be found `here <https://drive.google.com/drive/folders/15MDzLSSBNFNMDnj-dD6bBRWcC90k1kUj?usp=drive_link>`_

Beyond benchmark results, there are additional reasons to prefer :code:`experiment-design` over the listed libraries.
As of this writing, none of these libraries natively support:

- Correlated variables,
- Non-uniform distributions,
- Extending |LHS| by adding new samples while preserving |LHS| properties.
