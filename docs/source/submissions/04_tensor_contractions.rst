Tensor Contractions on GPUs
===========================

Task 1: Tiled Contraction Kernel Variants
-----------------------------------------

a) Dimension Classification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The dimensions for the einsum string ``eabklxy, ecklyz -> eabcxz`` are classified as follows:

- ``M``: ``a``, ``b``, ``x``
- ``N``: ``c``, ``z``
- ``K``: ``k``, ``l``, ``y``
- ``C``: ``e``

.. _task_1b:

b) cuTile Kernel: Sequentialize over ``k`` and ``l``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We create three tensors ``A``, ``B``, and ``C`` with random dimension sizes and values. 
For that we use a ``tensor_initialization()`` function that creates these tensors for us:

.. literalinclude:: ../../../src/assignments/04_assignment/utils.py
    :language: py
    :linenos:
    :lines: 13-48
    :caption: `Function to initialize three random tensors`

Inside this function we have a check that prevents us to create three tensors that exceed the memory limit of ``32 GiB``.
If our tensors exceed the limit, we simply generate a new random dimension size based on the current value. 
As this procedure is wrapped inside a ``while``-loop we guarantee to be within the memory limit. 

The actual task was solved by creating a ``1D``-grid over the whole ``C`` tensor.

.. literalinclude:: ../../../src/assignments/04_assignment/task-01.py
    :language: py
    :linenos:
    :lines: 54-61
    :caption: `Grid creation and kernel launch`

Based on this grid it was then possible to retrieve the block IDs of the respective dimensions involved in the computation.

.. _1b_bid:

.. literalinclude:: ../../../src/assignments/04_assignment/task-01.py
    :language: py
    :linenos:
    :lines: 17-33
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
    :lines: 37-51
    :caption: `GEMM computation`

Due to the reason that the ranks of tensor ``A`` and tensor ``B`` (and ``C``) where not matching we decided to create an accumulation matrix ``acc`` of the shape ``(tM, tN)``.
The next thing we did was to set up the sequential ``K``-type loops over ``k``, ``l``, and ``y``, while ``y`` is only needed for the case that :math:`tK < y`.
As we had to create the accumulation with a new shape, we were forced to also reshape the loaded tiles from tensor ``A`` and ``B``. 

After going through all of the loops sequentially we then reshape our accumulated tensor back to the original shape, in order to store the result in our ``C`` tensor.

c) cuTile Kernel: Sequentialize over ``k``, ``l`` and ``b``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To additionally sequentialize the we first changed the grid by extracting the ``b`` dimension from the calculation.

.. literalinclude:: ../../../src/assignments/04_assignment/task-01.py
    :language: py
    :linenos:
    :lines: 106-113
    :caption: `Grid creation and kernel launch`

Reducing the grid dimension also means that we need to change the index calculation, as we don't need to calculate the ``b`` block ID any more.

.. literalinclude:: ../../../src/assignments/04_assignment/task-01.py
    :language: py
    :linenos:
    :lines: 71-84
    :caption: `Block ID computation`

In regards to the computation we add a new loop for the ``b`` dimension and also move the creation of the ``acc`` tile inside that loop.

.. literalinclude:: ../../../src/assignments/04_assignment/task-01.py
    :language: py
    :linenos:
    :lines: 86-101
    :caption: `GEMM computation`

To find one dimension configuration where the kernel from :ref:`Task 1b) <task_1b>` performed better, we did not need to experiment too much.
One possible configuration is ``(e=8, a=8, b=16, c=8, k=8, l=16, x=16, y=16, z=16)``.

On the other hand to find a kernel where the implementation of Task 1c) was better, was harder. 
Our approach was too equally distribute the workload by sharing the output over the 48 streaming multiprocessors.
Further, we made sure that the ``b`` dimension is odd. 
Finally we found a dimension configuration where the kernel from Task 1c) was slightly better: ``(e=16, a=8, b=5, c=6, k=8, l=8, x=8, y=8, z=8)``.

.. _1c_bench:

.. code-block:: text
    
    [expected 1b > 1c > 1d]
      total memory: 0.07421875GiB
      grid: 1b=32768 blocks, 1c=2048 blocks, 1d=32768 blocks
      tile: tM=8, tN=8, tK=8
      1b=46.091ms  1c=47.734ms  1d=213.975ms
      Ranking: 1b > 1c > 1d
    [expected 1c > 1b]
      total memory: 0.00653076171875GiB
      grid: 1b=15360 blocks, 1c=3072 blocks, 1d=15360 blocks
      tile: tM=4, tN=4, tK=4
      1b=5.898ms  1c=5.863ms  1d=8.908ms
      Ranking: 1c > 1b > 1d

d) cuTile Kernel: GEMM Dimensions ``xyzl``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order for us to use the dimensions ``xyzl`` as GEMM dimensions, we switch back to the grid from :ref:`Task 1b) <task_1b>`.
Further, we prepare the tensors ``A`` and ``B``. 

For tensor ``A`` we perform a permutation to change the dimensionality to ``eabklxy -> eabkxly`` and then we reshape the tensor accordingly.
On the other hand, tensor ``B`` can be directly reshaped. 

.. literalinclude:: ../../../src/assignments/04_assignment/task-01.py
    :language: py
    :linenos:
    :lines: 163-183
    :caption: `Permute and reshape for A and B`

We also switch back to the :ref:`block ID computation <1b_bid>` from Task 1b). 
What changes is the way our kernel computes, as we combine the ``l`` and ``y`` dimension to ``ly``.
By combining them we essentially compute a ``xyzl`` GEMM.

.. literalinclude:: ../../../src/assignments/04_assignment/task-01.py
    :language: py
    :linenos:
    :lines: 141-155
    :caption: `xyzl GEMM`

Similarly to `Task 1c) <1c_bench>` we could use the same dimension configuration to outperform the kernel from Task 1d) with the kernel from Task 1b).

To find a dimension configuration where the kernel from Task 1d) outperforms the kernel from Task 1b) we reduced the size of the kernel ``xyzl`` compared to the remaining dimensions.
The idea was to make the workload big enough for the kernel from Task 1d) to be equally utilized as the kernel from Task 1b), while performing less kernel calls.

.. code-block:: text
    
    [expected 1b > 1c > 1d]
      total memory: 0.07421875GiB
      grid: 1b=32768 blocks, 1c=2048 blocks, 1d=32768 blocks
      tile: tM=8, tN=8, tK=8
      1b=46.091ms  1c=47.734ms  1d=213.975ms
      Ranking: 1b > 1c > 1d
    [expected 1d > 1b]
      total memory: 0.009567663073539734GiB
      grid: 1b=151200 blocks, 1c=15120 blocks, 1d=151200 blocks
      tile: tM=2, tN=2, tK=2
      1b=28.395ms  1c=28.570ms  1d=25.887ms
      Ranking: 1d > 1b > 1c

e) cuTile Kernel: 3D GEMM Kernel using ``exyz``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Creating a 3D ``exyz`` GEMM kernel we first permuted our tensors by moving the ``e`` dimension closer to the right:

.. literalinclude:: ../../../src/assignments/04_assignment/task-01.py
    :language: py
    :linenos:
    :lines: 232-238
    :caption: `Tensor permutations`

Secondly, we introduced a new tile size called ``tC`` which is used for tiling the ``e`` dimension.
Therefore, our grid changed as well.

.. literalinclude:: ../../../src/assignments/04_assignment/task-01.py
    :language: py
    :linenos:
    :lines: 240-246
    :caption: `Grid computation and kernel launch`

Inside the kernel we adjust the block ID calculation slightly, because of the positioning of dimension ``e`` and the new tile shape ``tC``. 

.. literalinclude:: ../../../src/assignments/04_assignment/task-01.py
    :language: py
    :linenos:
    :lines: 193-209
    :caption: `Block ID computation`

To perform a correct 3D GEMM we also change the indices and shapes for the tiles of ``A``, ``B``, ``acc`` and ``C``.

.. literalinclude:: ../../../src/assignments/04_assignment/task-01.py
    :language: py
    :linenos:
    :lines: 211-226
    :caption: `exyz GEMM`

The last thing is to bring the output matrix ``C`` back into its normal shape. 

.. literalinclude:: ../../../src/assignments/04_assignment/task-01.py
    :language: py
    :linenos:
    :lines: 248-249
    :caption: `Restore shape of C`
