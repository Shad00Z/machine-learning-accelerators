Contraction Interface and L2 Optimization
=========================================

.. _config:

Task 1: Config Class
--------------------

For our Config object we first defined enumeration types using ``enum.Enum`` for all of the configuration parameters.

.. literalinclude:: ../../../src/assignments/assignment_05/config.py
    :language: py
    :linenos:
    :lines: 9-44
    :caption: `Configuration parameter definitions`

After the definitions of the parameters we proceed with the ``Config`` class itself.
The ``Config`` class has 8 attributes: 

.. literalinclude:: ../../../src/assignments/assignment_05/config.py
    :language: py
    :linenos:
    :lines: 77-86
    :caption: `Config attributes`

For debugging purposes we also added a python ``__repr__`` function, which helps in preserving all the necessary information about the config object: 

.. literalinclude:: ../../../src/assignments/assignment_05/config.py
    :language: py
    :linenos:
    :lines: 88-102
    :caption: `__repr__ function`

.. _task_2:

Task 2: Generating a Basic Config
---------------------------------

After creating our basic setup with the configuration parameters and the ``Config`` itself, we can now create a `generate_config` function. 
This ``generate_config`` function takes an einsum string and a list of shapes for the input tensors and returns a basic `Config`. 

We broke the generation down to a four step procedure. 
The first step is to parse the einsum string to get output and input indices. 

.. literalinclude:: ../../../src/assignments/assignment_05/config.py
    :language: py
    :linenos:
    :lines: 190-196
    :caption: `Step 1: Parse the einsum string`

We make sure that the list of indices we retrieve from the einsum string matches with the input shapes that were handed to the `generate_config` function.

After the initial verification, we collect the dimension sizes and check for their consistency.

.. literalinclude:: ../../../src/assignments/assignment_05/config.py
    :language: py
    :linenos:
    :lines: 199-216
    :caption: `Step 2: Dimension index collection`

The collection is performed tensor by tensor. 
First, each tensor's indices and shapes are compared to rule out rank mismatches. 
Then, the single indices of the tensor are either added to the ``dim_sizes`` dictionary or an exception is raised, if an index already exists and it conflicts with the already existing dimension size.

With these ``dim_sizes`` we build the per-dimension attribute lists.

.. literalinclude:: ../../../src/assignments/assignment_05/config.py
    :language: py
    :linenos:
    :lines: 218-225
    :caption: `Step 3: Attribute list for dimensions`

To create an attribute list we iterate over our ``dim_sizes`` dictionary and classify each dimension. 

.. literalinclude:: ../../../src/assignments/assignment_05/config.py
    :language: py
    :linenos:
    :lines: 123-147
    :caption: `Dimension classification`

Dimension classification means we check in which tensors a dimension appears and according to that we classify the dimension.
For this third step, we also assign each dimension the ``SEQ`` execution type. 

In the fourth and final step we compute the strides for each tensor. 

.. literalinclude:: ../../../src/assignments/assignment_05/config.py
    :language: py
    :linenos:
    :lines: 228-232
    :caption: `Step 4: Stride computation`

First of all, we assign each dimension that is not appearing inside a tensor the value 0.
The calculation for the remaining dimensions is pretty straightforward. 

.. literalinclude:: ../../../src/assignments/assignment_05/config.py
    :language: py
    :linenos:
    :lines: 150-168
    :caption: `Stride multiplication`

We start with the rightmost index for a tensor and assign the value 1.
To calculate the strides we go from the rightmost to the leftmost index for each tensor.
Afterwards we multiply the size of the respective tensor (e.g. the rightmost) with the current stride and thereby receive the stride for next dimension index.

.. _optimizer:

Task 3: Optimizer Class
-----------------------

a) Split Dimensions
^^^^^^^^^^^^^^^^^^^

For the ``split_dim`` function we follow a number of steps to guarantee that the dimension split is performed correctly.

First we verify that the dimension that is supposed to be split is a valid dimension of our configuration and that :math:`outer\_size * inner\_size` equals the original size.

.. literalinclude:: ../../../src/assignments/assignment_05/optimizer.py
    :language: py
    :linenos:
    :lines: 19-28
    :caption: `Split dimension: Dimension verification`

After the verification we replace the original dimension entry in the config object with the new ``inner`` and ``outer`` dimension.

.. literalinclude:: ../../../src/assignments/assignment_05/optimizer.py
    :language: py
    :linenos:
    :lines: 33-42
    :caption: `Split dimension: Replacing the original entry`

The last step is to calculate the strides for the new dimensions and store them in the config object. 

.. literalinclude:: ../../../src/assignments/assignment_05/optimizer.py
    :language: py
    :linenos:
    :lines: 44-53
    :caption: `Split dimension: Stride calculation`

b) Fuse Dimensions
^^^^^^^^^^^^^^^^^^

For the ``fuse_dim`` function we follow a similar approach. 

As a first step, we verify again that the dimensions that are supposed to be fused are part of our config object. 

.. literalinclude:: ../../../src/assignments/assignment_05/optimizer.py
    :language: py
    :linenos:
    :lines: 70-78
    :caption: `Fuse dimensions: Dimension verification`

Next we check if the dimensions are placed contiguously in memory. 

.. literalinclude:: ../../../src/assignments/assignment_05/optimizer.py
    :language: py
    :linenos:
    :lines: 80-106
    :caption: `Fuse dimensions: Check continuity`

After this check we calculate the new stride for the fused dimension. 

.. literalinclude:: ../../../src/assignments/assignment_05/optimizer.py
    :language: py
    :linenos:
    :lines: 108-122
    :caption: `Fuse dimensions: Stride calculation`

Finally we store the fused dimension and the stride at the respective position and remove the previous indices. 

.. literalinclude:: ../../../src/assignments/assignment_05/optimizer.py
    :language: py
    :linenos:
    :lines: 124-129
    :caption: `Fuse dimensions: Storing the new dimension`

c) Permute Dimensions
^^^^^^^^^^^^^^^^^^^^^

Our ``permute_dims`` function follows the syntax of the ``torch.permute`` function. 
That means the ``i-th`` entry of the result comes from the position ``permutation[i]`` in the original.

.. literalinclude:: ../../../src/assignments/assignment_05/optimizer.py
    :language: py
    :linenos:
    :lines: 135-154
    :caption: `Permute dimensions`

d) Execution
^^^^^^^^^^^^

The ``make_executable`` function assigns execution types and permutes the dimensions of the config object. 
We distinguish three types of execution types: ``PRIM``, ``PAR``, and ``SEQ``. 

By default each dimension is assigned the ``SEQ`` execution type. 
For the ``PRIM`` execution type we select the dimension with the smallest non-zero stride for each tensor. 
We select a dimension for each of the dimension types ``M``, ``N``, and ``K``. 

.. literalinclude:: ../../../src/assignments/assignment_05/optimizer.py
    :language: py
    :linenos:
    :lines: 175-197
    :caption: `Exec types: PRIM selection`

After assigning the ``PRIM`` execution types, we once again loop over all dimensions and change all dimensions to a ``PAR`` execution type that aren't of dimension type ``K``.

.. literalinclude:: ../../../src/assignments/assignment_05/optimizer.py
    :language: py
    :linenos:
    :lines: 199-206
    :caption: `Exec types: PAR assignments`

The last step is to move the ``SEQ`` dimensions to the leftmost positions and the ``PRIM`` dimensions to the rightmost positions. 
We verify the new execution types with the :ref:`verify <verification>` function.

.. literalinclude:: ../../../src/assignments/assignment_05/optimizer.py
    :language: py
    :linenos:
    :lines: 208-215
    :caption: `Exec types: SEQ and PRIM reordering`

.. _verification:

A tensor: K rightmost
B tensor: N rightmost
C tensor: M rightmost

e) Verification
^^^^^^^^^^^^^^^

The ``verify`` function checks the executability of the config object. 
We perform several checks in order to guarantee that a config object is executable.

First, we verify that no dimension of type ``K`` has the ``PAR`` execution type assigned. 

.. literalinclude:: ../../../src/assignments/assignment_05/optimizer.py
    :language: py
    :linenos:
    :lines: 228-233
    :caption: `Verify: No PAR assignments for K types`

Then we verify the dimension order. 

.. literalinclude:: ../../../src/assignments/assignment_05/optimizer.py
    :language: py
    :linenos:
    :lines: 235-267
    :caption: `Verify: SEQ, PAR, PRIM ordering`

The next step is to verify that each tensor has at least 2 ``PRIM`` dimensions (``M, K``, ``K, N``, ``M, N``).
We also check their continuity in memory.

.. literalinclude:: ../../../src/assignments/assignment_05/optimizer.py
    :language: py
    :linenos:
    :lines: 269-283
    :caption: `Verify: Tensor PRIM dimensions`

Task 4: L2-Optimized Batched Contraction
----------------------------------------

a) Initial Config
^^^^^^^^^^^^^^^^^

For the batched matrix multiplication ``cmk, ckn -> cmn`` our ``generate_config`` function from :ref:`Task 2 <task_2>` produces the following config:

.. code-block:: text

    Config(
        data_type  = DataType.FLOAT16
        prim_main  = PrimType.GEMM
        prim_last  = LastType.NONE
        prim_first = FirstType.ZERO
        dim_types  = ['C', 'M', 'K', 'N']
        exec_types = ['SEQ', 'SEQ', 'SEQ', 'SEQ']
        dim_sizes  = [4, 4096, 4096, 4096]
        strides[0] = [16777216, 4096, 1, 0]
        strides[1] = [16777216, 0, 4096, 1]
        strides[2] = [16777216, 4096, 0, 1]
    )

b) Optimized Config
^^^^^^^^^^^^^^^^^^^

To optimize the initial config object we made five transformations. 

First, we split the ``N`` dimension into ``n1=4`` and ``n2=1024``.
Second, we split the ``M`` dimension into ``m1=4`` and ``m2=1024``.
To incorporate the tile sizes directly into the configuration, we additionally split the three PRIM dimensions:
the ``M`` inner dimension (1024) into ``(16, tM=64)``, the ``K`` dimension (4096) into ``(64, tK=64)``, and the ``N`` inner dimension (1024) into ``(16, tN=64)``.
After calling ``make_executable()``, the PRIM dimensions carry exactly the tile sizes, so no separate heuristic is needed to derive them.

We decided to go with ``n2=m2=1024`` because with these dimensions we use the L2 cache optimally. 
After this dimension splitting we have ``4194304 bytes`` left in the L2 cache. 
With a bigger dimension size we would not be able to keep the whole computation within the L2 cache.

.. code-block:: text

    Config(
      data_type  = DataType.FLOAT16
      prim_main  = PrimType.GEMM
      prim_last  = LastType.NONE
      prim_first = FirstType.ZERO
      dim_types  = ['C', 'M', 'M', 'N', 'N', 'K', 'M', 'K', 'N']
      exec_types = ['PAR', 'PAR', 'PAR', 'PAR', 'PAR', 'SEQ', 'PRIM', 'PRIM', 'PRIM']
      dim_sizes  = [4, 4, 16, 4, 16, 64, 64, 64, 64]
      strides[0] = [16777216, 4194304, 262144, 0, 0, 64, 4096, 1, 0]
      strides[1] = [16777216, 0, 0, 1024, 64, 262144, 0, 4096, 1]
      strides[2] = [16777216, 4194304, 262144, 1024, 64, 0, 4096, 0, 1]
    )

c) Optimized Kernel
^^^^^^^^^^^^^^^^^^^

The kernel that executes this optimized config object for the initial batched matrix multiplication ``cmk, ckn -> cmn`` consists of two parts:

1. a launch function,
2. the kernel itself.

The launch kernel creates the grid according to the output tensor ``c * m1 * n1 * m2 * n2`` and forwards all the data to the kernel:

.. literalinclude:: ../../../src/assignments/assignment_05/task-04.py
    :language: py
    :linenos:
    :lines: 105-117
    :caption: `Launch function`

The kernel itself calculates the block IDs according to the shape of the output tensor. 

.. literalinclude:: ../../../src/assignments/assignment_05/task-04.py
    :language: py
    :linenos:
    :lines: 73-87
    :caption: `BID calculation`

Then we load the tiles and calculate our batched matrix multiplication.

.. literalinclude:: ../../../src/assignments/assignment_05/task-04.py
    :language: py
    :linenos:
    :lines: 89-99
    :caption: `Batched Matrix Multiplication`

The last thing we do is to compare the correctnes of our kernel against ``torch.allclose``. 

d) TFLOP Benchmarking
^^^^^^^^^^^^^^^^^^^^^

We compare the implementation of our optimized batched matrix multiplication with a simple batched matrix multiplication kernel on the original tensors. 

As the number of ``TFLOPs`` highly depends on the tile sizes, we created several benchmarks comparing different sizes. 

.. list-table::
    :widths: 50 50
    :header-rows: 1
    
    * - Optimized Kernel 
      - Baseline Kernel

    * - .. image:: ../../../src/assignments/assignment_05/resources-05/task4_k=16_heatmap.png
           :width: 100%

      - .. image:: ../../../src/assignments/assignment_05/resources-05/task4_k=16_heatmap_ref.png
           :width: 100%

    * - .. image:: ../../../src/assignments/assignment_05/resources-05/task4_k=32_heatmap.png
           :width: 100%

      - .. image:: ../../../src/assignments/assignment_05/resources-05/task4_k=32_heatmap_ref.png
           :width: 100%

    * - .. image:: ../../../src/assignments/assignment_05/resources-05/task4_k=64_heatmap.png
           :width: 100%

      - .. image:: ../../../src/assignments/assignment_05/resources-05/task4_k=64_heatmap_ref.png
           :width: 100%

    * - .. image:: ../../../src/assignments/assignment_05/resources-05/task4_k=128_heatmap.png
           :width: 100%

      - .. image:: ../../../src/assignments/assignment_05/resources-05/task4_k=128_heatmap_ref.png
           :width: 100%

Comparing these results, it is obvious that the optimized kernel achieves a significantly higher number of ``TFLOPs`` than the baseline kernel.
At the peak (tile sizes ``tM=128, tN=64, tK=128``), the optimized kernel achieves ``60.7 TFLOPs`` while the baseline kernel only achieves ``28.1 TFLOPs``. 
This shows the importance of well thought out memory usage. 
