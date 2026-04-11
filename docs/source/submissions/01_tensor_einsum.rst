Tensor and Einsum
=================

.. _dot-product:

Task 1: Dot Product
-------------------

To calculate the dot product of two vectors for variable length, one simply needs to loop over the single shared dimension.

.. literalinclude:: ../../../src/assignments/01_assignment/assignment_01.py
    :language: py
    :linenos:
    :lines: 16-17
    :caption: `Single loop over the shared dimension`

.. _mat-mat-mul:

Task 2: Matrix-Matrix Multiplication
------------------------------------

Calculating a matrix-matrix multiplication simply adds another dimension to the calculation.
Therefore, calculating the matrix-matrix multiplication using loops can be simply done by adding two more loops for the ``m`` and ``n`` dimension.

.. literalinclude:: ../../../src/assignments/01_assignment/assignment_01.py
    :language: py
    :linenos:
    :lines: 36-39
    :caption: `Three loops to calculate the result of the matrix-matrix multiplication.`

Reusing the dot product calculation from :ref:`Task 1<dot-product>` can be achieved by replacing the inner most loop by the function call to the ``dot_product`` function.

.. literalinclude:: ../../../src/assignments/01_assignment/assignment_01.py
    :language: py
    :linenos:
    :lines: 54-57
    :caption: `Two loops iterating over the dot_product function call.`

Task 3: Einsum
--------------

In order to properly calculate the einsum expression, we need one loop for each dimension.
The loops have been ordered accordingly to their indices. The further an index is on the right, the further it is nested.

.. literalinclude:: ../../../src/assignments/01_assignment/assignment_01.py
    :language: py
    :linenos:
    :lines: 77-84
    :caption: `Several nested loops to calculate the einsum expression.`

The nesting depth of the first calculation can be reduced by invoking the ``matmul_dot`` function from :ref:`Task 2<mat-mat-mul>`.
We selected the inner most dimensions of the matrices ``A`` and ``B`` to pass to the ``matmul_dot`` function.

.. literalinclude:: ../../../src/assignments/01_assignment/assignment_01.py
    :language: py
    :linenos:
    :lines: 101-105
    :caption: `Calculating the einsum expression by calling the matmul_dot function.`
