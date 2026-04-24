Matrix Multiplication with cuTile
=================================

Task 1: FP32 vs FP16 Performance
--------------------------------

Our ``FP16`` and ``FP32`` kernels are implemented equally:

.. literalinclude:: ../../../src/assignments/03_assignment/task-01.py
    :language: py
    :linenos:
    :lines: 14-24
    :caption: `FP16 kernel for matrix multiplication`

After implementing both, the ``FP16`` and the ``FP32`` kernel, we measured the following execution times:

.. code-block:: text

    Benchmark (ms per launch):
        FP16 time: 0.028004568445800248
        FP32 time: 1.7323399588076567

These measurements show that the speedup of the ``kernel_fp16`` over ``kernel_fp32`` is about ``87``.
This shows that performing calculations at a lower precision results in significant performance gains.

Task 2: Simple Matrix Multiplication
------------------------------------

Implementing the matrix multiplication kernel using ``ct.mma``, we followed the provided instructions. 

1. The two input matrices ``A`` and ``B`` are created with dimensions in the range from ``1`` to ``4096``.
2. Our grid depends on the amount of tiles for the ``n`` and ``m`` dimension.

.. literalinclude:: ../../../src/assignments/03_assignment/task-02.py
    :language: py
    :linenos:
    :lines: 58
    :caption: `1d grid`

3. The kernel itself uses the block id to calculate the ``row`` and ``column`` according to a row-major format.
   By adding the padding mode to the ``ct.load`` operations, we are also handling matrix sizes that are not powers of 2.

.. literalinclude:: ../../../src/assignments/03_assignment/task-02.py
    :language: py
    :linenos:
    :lines: 18-35
    :caption: `Simple matrix multiplication kernel`
