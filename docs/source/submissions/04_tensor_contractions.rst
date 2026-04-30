Tensor Contractions on GPUs
===========================

Task 1: Tiled Contraction Kernel Variants
-----------------------------------------

Dimension Classification
^^^^^^^^^^^^^^^^^^^^^^^^

The dimensions for the einsum string ``eabklxy, ecklyz -> eabcxz`` are classified as follows:

- ``M``: ``a``, ``b``, ``x``
- ``N``: ``c``, ``z``
- ``K``: ``k``, ``l``, ``y``
- ``C``: ``e``

cuTile Kernel: Sequentialize over ``k`` and ``l``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We create three tensors ``A``, ``B``, and ``C`` with random dimension sizes and values. 
For that we use a ``tensor_initialization()`` function that creates these tensors for us:

.. literalinclude:: ../../../src/assignments/04_assignment/utils.py
    :language: py
    :linenos:
    :lines: 12-47
    :caption: `Function to initialize three random tensors`

Inside this function we have a check that prevents us to create three tensors that exceed the memory limit of ``32 GiB``.
If our tensors exceed the limit, we simply generate a new random dimension size based on the current value. 
As this procedure is wrapped inside a ``while``-loop we guarantee to be within the memory limit. 

The actual task was solved by creating a ``1D``-grid over the whole ``C`` tensor.

.. literalinclude:: ../../../src/assignments/04_assignment/task-01.py
    :language: py
    :linenos:
    :lines: 67-73
    :caption: `Grid creation and kernel launch`

Based on this grid it was then possible to retrieve the block IDs of the respective dimensions involved in the computation.

.. literalinclude:: ../../../src/assignments/04_assignment/task-01.py
    :language: py
    :linenos:
    :lines: 19-35
    :caption: `Block ID computation`

The block ID compuation follows a simple principle. 
As the fast dimension is the rightmost dimension in the einsum string, we start with index ``z`` and compute the block ID of ``z`` with the retrieved ``bid = ct.bid(0)``.
After computing the index, we reduce ``bid`` by the number of tiles we have for dimension ``z``.

This concludes one iteration. 
We repeat this for every single index, going from left to right:

1. ``z``
2. ``x``
3. ``c``
4. ``b``
5. ``a``
6. ``e``

After retrieving these indices we set up our **GEMM**:

.. literalinclude:: ../../../src/assignments/04_assignment/task-01.py
    :language: py
    :linenos:
    :lines: 37-52
    :caption: `GEMM computation`

Due to the reason that the ranks of tensor ``A`` and tensor ``B`` (and ``C``) where not matching we decided to create an accumulation matrix ``acc`` of the shape ``(tM, tN)``.
The next thing we did was to set up the sequential ``K``-type loops over ``k``, ``l``, and ``y``, while ``y`` is only needed for the case that :math:`tK < y`.
As we had to create the accumulation with a new shape, we were forced to also reshape the loaded tiles from tensor ``A`` and ``B``. 

After sequentially going through all of the loops we then also reshape our accumulated tensor back to the original shape, in order to store the result in our ``C`` tensor.
