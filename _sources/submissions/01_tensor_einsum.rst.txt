Tensor and Einsum
=================

.. _dot-product:

Task 1: Dot Product
-------------------

To calculate the dot product of two vectors for variable length, one simply needs to loop over the single shared dimension.

.. literalinclude:: ../../../src/assignments/assignment_01/assignment_01.py
    :language: py
    :linenos:
    :lines: 16-17
    :caption: `Single loop over the shared dimension`

.. _mat-mat-mul:

Task 2: Matrix-Matrix Multiplication
------------------------------------

Calculating a matrix-matrix multiplication simply adds another dimension to the calculation.
Therefore, calculating the matrix-matrix multiplication using loops can be simply done by adding two more loops for the ``m`` and ``n`` dimensions.

.. literalinclude:: ../../../src/assignments/assignment_01/assignment_01.py
    :language: py
    :linenos:
    :lines: 36-39
    :caption: `Three loops to calculate the result of the matrix-matrix multiplication.`

Reusing the dot product calculation from :ref:`Task 1<dot-product>` can be achieved by replacing the innermost loop by the function call to the ``dot_product`` function.

.. literalinclude:: ../../../src/assignments/assignment_01/assignment_01.py
    :language: py
    :linenos:
    :lines: 54-57
    :caption: `Two loops iterating over the dot_product function call.`

Task 3: Einsum
--------------

In order to properly calculate the einsum expression, we need one loop for each dimension.
The loops have been ordered according to their indices. The further an index is on the right, the further it is nested.

.. literalinclude:: ../../../src/assignments/assignment_01/assignment_01.py
    :language: py
    :linenos:
    :lines: 77-84
    :caption: `Several nested loops to calculate the einsum expression.`

The nesting depth of the first calculation can be reduced by invoking the ``matmul_dot`` function from :ref:`Task 2<mat-mat-mul>`.
We selected the innermost dimensions of the matrices ``A`` and ``B`` to pass to the ``matmul_dot`` function.

.. literalinclude:: ../../../src/assignments/assignment_01/assignment_01.py
    :language: py
    :linenos:
    :lines: 101-105
    :caption: `Calculating the einsum expression by calling the matmul_dot function.`

Task 4: Tensor Permutation and Reshaping
----------------------------------------

The experiments for the tensor permutation and reshaping are located in the ``permute_reshape.py`` file.

Assume we initially have the following tensor:

.. code-block:: text

    # Initial tensor
    tensor([[ 1,  2,  3,  4],
            [ 5,  6,  7,  8],
            [ 9, 10, 11, 12],
            [13, 14, 15, 16]])

    # Shape
    torch.Size([4, 4])

    # Stride
    (4, 1)

    # Contiguity
    True

Permutation
^^^^^^^^^^^

`torch.permute <https://docs.pytorch.org/docs/stable/generated/torch.permute.html#torch-permute>`_ takes two input parameters:

1. the input tensor and
2. the permuted dimensions.

This function returns a view of the original tensor with the permuted dimensions.
Alternatively, one can also call ``<tensor>.permute(<dims>)``.

After applying the ``torch.permute(1, 0)`` function, the initial tensor changes to:

.. code-block:: text

    # Tensor after permute
    tensor([[ 1,  5,  9, 13],
            [ 2,  6, 10, 14],
            [ 3,  7, 11, 15],
            [ 4,  8, 12, 16]])

    # Shape
    torch.Size([4, 4])

    # Stride
    (1, 4)

    # Contiguity
    False

This shows that the memory layout **does** change after applying the ``torch.permute`` function.
Furthermore, the resulting tensor is **no** longer stored continuously in memory.

Reshape
^^^^^^^

`torch.reshape <https://docs.pytorch.org/docs/stable/generated/torch.reshape.html#torch-reshape>`_ takes two input parameters:

1. the input tensor and
2. the new shape.

This function returns a tensor with the same data and number of elements as the original input tensor, simply with a new shape.
If possible, the returned tensor comes in the shape of a view, otherwise it will be a copy.
For example, contiguous inputs and inputs with compatible strides are reshaped **without** copying.

In comparison, if we apply the ``torch.reshape(1, -1)`` function, the tensor looks like:

.. code-block:: text

    # Tensor after reshape
    tensor([[ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16]])

    # Shape
    torch.Size([1, 16])

    # Stride
    (16, 1)

    # Contiguity
    True

This shows that the order of the tensor elements is the same as the initial tensor.
Therefore, the memory layout does **not** change after this ``torch.reshape`` function call.

View
^^^^

Depending on the contiguity of the given tensor, the result of the ``torch.view`` operation differs from ``torch.reshape``.
We experimented with two different parameter assignments.
If we apply ``torch.reshape(1, -1)`` and ``torch.view(1, -1)`` to the same initial tensor, everything works the same.

However, if we first transpose the tensor using ``torch.transpose(0, 1)``, we can still apply the ``torch.reshape(1, -1)`` operation, but we can't apply the ``torch.view(1, -1)`` operation.

.. code-block:: text

    # Tensor after transpose
    tensor([[ 1,  5,  9, 13],
            [ 2,  6, 10, 14],
            [ 3,  7, 11, 15],
            [ 4,  8, 12, 16]])

    # Contiguity
    Tensor contiguity: False

.. code-block:: text

    # Tensor after transpose and reshape
    tensor([[ 1,  5,  9, 13,  2,  6, 10, 14,  3,  7, 11, 15,  4,  8, 12, 16]])

    # Shape
    torch.Size([1, 16])

    # Stride
    (16, 1)

    # Contiguity
    True

It is noticeable that the ``torch.reshape`` operation can be applied to non-contiguous tensors.
Furthermore, after applying ``torch.reshape``, the memory layout is contiguous again.

The reason why a permuted tensor can be reshaped is that the ``torch.reshape`` operation is not influenced by the memory layout of the given tensor.
As the description in torch already mentions, ``torch.reshape`` can handle tensors differently.
In the case of a **permuted** tensor, the reshape operation creates a **copy** of the permuted tensor.
In the case of a **freshly** created tensor, the reshape operation does **not** create a copy but rather creates a **view** of the freshly created tensor.
