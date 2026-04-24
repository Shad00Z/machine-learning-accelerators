Matrix Multiplication with cuTile
=================================

Task 1: FP32 vs FP16 Performance
--------------------------------

Our ``FP16`` and ``FP32`` kernels are implemented equally:

.. literalinclude:: ../../../src/assignments/03_assignment/task-01.py
    :language: py
    :linenos:
    :lines: 15-24
    :caption: `FP16 kernel for matrix multiplication`

After implementing both, the ``FP16`` and the ``FP32`` kernel, we measured the following execution times:

.. code-block:: text

    Benchmark (ms per launch):
        FP16 time: {} 0.028004568445800248
        FP32 time: {} 1.7323399588076567

These measurements show that the speedup of the ``kernel_fp16`` over ``kernel_fp32`` is about ``87``.
This shows that performing calculations at a lower precision results in significant performance gains.
