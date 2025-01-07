.. |CDF| replace:: :abbr:`CDF (Cumulative Distribution Function)`
.. |DoE| replace:: :abbr:`DoE (Design of Experiments)`
.. |LHS| replace:: :abbr:`LHS (Latin Hypercube Sampling)`

Experiment Design via Orthogonal Sampling
'''''''''''''''''''''''''''''''''''''''''

Assume that we want to conduct experiments and we have a total control on which values the input parameters will take.
There are a number of approaches to select values, from

 - `factorial designs <https://en.wikipedia.org/wiki/Factorial_experiment>`_,
 - `central composite design <https://en.wikipedia.org/wiki/Central_composite_design>`_,
 - `A-, C-, D-, ... optimal designs <https://en.wikipedia.org/wiki/Optimal_experimental_design>`_,

and so on. Although all of these designs have various nice properties such as being space-filling or efficient with
respect to some optimization criteria, they do not take the parameter uncertainty into account. However, modelling the
uncertainty may be required by some tasks, e.g. when estimating statistical moments from the sample. Orthogonal design
allows us to model the parameter uncertainties while providing experiment designs with higher quality compared to
random sampling with respect to space-filling properties. Since orthogonal sampling is a generalization of Latin
hypercube sampling ( |LHS| ) to non-uniform variables, we will start by describing how an experiment can be designed
using |LHS| and why it is superior to random sampling.

Latin hypercube sampling
------------------------

|LHS| is a generalization of the Latin sphere to higher dimensional spaces, first proposed by
`McKay et al. (1979) <https://www.researchgate.net/publication/235709905_A_Comparison_of_Three_Methods_for_Selecting_Vales_of_Input_Variables_in_the_Analysis_of_Output_From_a_Computer_Code>`_.
We will need three steps to generate an |LHS|. For visualization purposes, we will be using a two dimensional space
with the bounds :math:`[0, 1]^2`. Before generating the design, we need to decide how many samples we will need. For now
let us create 8 samples. First, we partition the space into small squares (or hypercubes, if we had more than two
dimensions), such that each dimension is partitioned into 8 parts. We will be calling these hypercubes bins from here on out.
We can visualize this as follows:


.. code:: python

    import matplotlib.pyplot as plt
    import numpy as np

    bin_edges = np.linspace(0, 1, 9)  # we need 9 lines to represents 8 bins
    plt.figure()
    for x in bin_edges:
        plt.plot([x, x], [0, 1], c="k")
        plt.plot([0, 1], [x, x], c="k")

.. image:: images/os_lhs_grid.png
    :align: center

Next, we place each sample such that each bin is occupied only ones in each direction. This is quite easy to implement,
but since we are show casing the capabilities of experiment-design, let us use it here.

.. code:: python

    from experiment_design import create_continuous_uniform_space, OrthogonalSamplingDesigner

    np.random.seed(1337)
    space = create_continuous_uniform_space([0., 0], [1., 1.])
    designer = OrthogonalSamplingDesigner(inter_bin_randomness=0.)
    doe = designer.design(space, sample_size=8, steps=1)
    plt.scatter(doe[:, 0], doe[:, 1], label="Init. design")

.. image:: images/os_lhs_init.png
    :align: center

There are a few important details in the above code so let's walk line by line. After the import, we first set a random
seed. This is important for the reproducibility. Given the same inputs and seed, we will always generate the same design
on the same machine. Next, we define a two dimensional parameter space (:class:`.ParameterSpace`)
within the bounds :math:`[0, 1]^2`. Note that in general, bounds do not have to be equal, they can be any finite number
as long as the lower bound at the index m representing the variable m is smaller than the upper bound at the index m.
Following, we initiate an :class:`.OrthogonalSamplingDesigner`
with the parameter :code:`inter_bin_randomness=0.`. This controls the randomness of the placement of samples within the
bins. A value of 0. places the samples exactly in the middle of the bins, whereas a value of 0.8 (default) would lead to
placing samples anywhere between :math:`[-0.4 \delta, 0.4 \delta]` within the bin, where :math:`\delta` is the bin size,
here :math:`1/8=0.125`. Finally, we generate a doe using only 1 step, i.e. skipping any optimization for now, that we
would do normally and plot the result.

Final step is not mandatory, but it improves the |DoE| quality a lot, as proposed by `Joseph et al. (2008) <https://www3.stat.sinica.edu.tw/statistica/oldpdf/A18n17.pdf>`_:
Optimize the samples using simulated annealing by switching the values of samples along each dimension. We will talk about
the optimization objectives later. Notice that any switches would not violate the |LHS| rules; each bin would still be
occupied only once. This is done automatically in experiment-design unless we turn it off as we did before. In order to
start from the same |DoE|, we set the same seed but use the default number of steps.


.. code:: python

    np.random.seed(1337)
    doe2 = designer.design(space, sample_size=8)
    plt.scatter(doe2[:, 0], doe2[:, 1], label="Final design")

.. image:: images/os_lhs_opt.png
    :align: center

Finally, let us also create some random samples just to use as a baseline. We can do this using experiment-design too.
Implicitly, there is also some search for the random sampler, where we evaluate the random |DoE| on the same set of
objectives as before and choose the one that achieves the best results. For the purposes of this document, we will
deactivate the optimization by setting :code:`steps=1` as we did before.

.. code:: python

    from experiment_design import RandomSamplingDesigner

    doe3 = RandomSamplingDesigner().design(space, sample_size=8, steps=1)
    plt.scatter(doe3[:, 0], doe3[:, 1], label="Random sampling")
    plt.legend()

.. image:: images/os_lhs_final.png
    :align: center

We can look at two metrics to evaluate the quality of the |DoE|; the minimum pairwise distance to evaluate its
space-filling properties as well as the correlation coefficient :math:`\rho` between the variables.

.. list-table::
    :header-rows: 1
    :align: center

    * - |DoE|
      - Min. distance
      - :math:`\rho`
    * - doe
      - 0.18
      - 0.00
    * - doe2
      - 0.35
      - 0.14
    * - doe3
      - 0.13
      - 0.19

Initial |LHS| has no correlation error, although the optimized |LHS| induces some correlation but it almost doubles the
minimum pairwise distance, filling the parameter space much better. This is partially due to the default objective we use
in experiment-design, where we put 9 times more emphasis on the space filling properties compared to the correlation error.
Nevertheless, as we will see later, we can change the weights we use arbitrarily and even supply a custom objective function.
In any case, both |LHS| designs achieve better metrics compared to random sampling.

Now that we have showcased how |LHS| samples are generated and that it may achieve a higher quality compared to random
sampling, let us talk about orthogonal sampling and why it is useful for statistical inference.


Orthogonal sampling
--------------------

It is straightforward to generalize |LHS| to orthogonal sampling, where we generate an |LHS| design in :math:`[0, 1]^d`,
in a d-dimensional parameter space, which we interpret as probabilities and use the inverse |CDF| functions of the
marginal variables to map them to actual values. Let us see this in action, again in a 2-dimensional space for
visualization purposes. Let us define two Gaussian variables :math:`X_1, X_2 \sim \mathcal{N}(2, 1)` with a mean of
2 and a variance of 1. Again, to generate 8 sample, we start by partitioning the probability space into 8, which yields
the same bounds as before. Next, we map them back to the original space. The code would look like this:


.. code:: python

    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats

    from experiment_design import ParameterSpace, OrthogonalSamplingDesigner

    space = ParameterSpace(variables=[stats.norm(2, 1) for _ in range(2)],
                           infinite_bound_probability_tolerance=2.5e-2)
    probability_bin_edges = np.linspace(0, 1, 9)
    # create an array of probabilities, where each column represents a variable
    probability_bin_edges = np.c_[probability_bin_edges, probability_bin_edges]
    bin_edges = space.value_of(probability_bin_edges)  # This internally calls scipy_distribution.ppf
    bin_edges[0] = space.lower_bound
    bin_edges[-1] = space.upper_bound

    plt.figure()
    for x in bin_edges:
        plt.plot([x, x], [bin_edges[0, 1], bin_edges[-1, 1]], c="k")
        plt.plot([bin_edges[0, 0], bin_edges[-1, 0]], [x, x], c="k")


.. image:: images/os_grid.png
    :align: center


.. code:: python

    np.random.seed(1337)
    designer = OrthogonalSamplingDesigner(inter_bin_randomness=0.)
    doe = designer.design(space, sample_size=8)
    plt.scatter(doe[:, 0], doe[:, 1])


.. image:: images/os_doe.png
    :align: center
