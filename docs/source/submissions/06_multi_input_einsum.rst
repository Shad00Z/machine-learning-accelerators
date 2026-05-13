Multi-Input Einsum Contraction
==============================

Task 1: PyTorch Reference Contraction
-------------------------------------

a) Classification
^^^^^^^^^^^^^^^^^

The tensor indices can be classified as:

- ``M``: ``a, c, x``
- ``N``: ``b, y``
- ``K``: ``s, p``

b) Einsum contraction ``tensor_abcyx``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The einsum expression can be written as: ``acspx,bspy->abcyx``.

.. code-block:: python
    :caption: Einsum contraction

    tensor_acspx_fp32 = torch.tensor(data['tensor_acspx'], device='cuda:0', dtype=torch.float32)
    tensor_bspy_fp32 = torch.tensor(data['tensor_bspy'], device='cuda:0', dtype=torch.float32)
    
    tensor_acspx_fp16 = torch.tensor(data['tensor_acspx'], device='cuda:0', dtype=torch.float16)
    tensor_bspy_fp16 = torch.tensor(data['tensor_bspy'], device='cuda:0', dtype=torch.float16)

    tensor_abcyx_fp32 = torch.einsum("acspx,bspy->abcyx", tensor_acspx_fp32, tensor_bspy_fp32).to(device='cpu')
    tensor_abcyx_fp16 = torch.einsum("acspx,bspy->abcyx", tensor_acspx_fp16, tensor_bspy_fp16).to(device='cpu')

c) Visualize ``FP16`` and ``FP32`` Results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The two calculated results for the einsum expression are simply passed to the ``plot_tensor()`` function. 

.. code-block:: python
    :caption: Creating visualizations

    plot_tensor(
        tensor_abcyx_fp32,
        path=f'{path}/results/torch_32.png',
        title='Lightfield Tensorring Decomposition - All Ranks: 64 - PyTorch - FP32'
    )
    
    plot_tensor(
        tensor_abcyx_fp16,
        path=f'{path}/results/torch_16.png',
        title='Lightfield Tensorring Decomposition - All Ranks: 64 - PyTorch - FP16'
    )

The visualization that is produced is a bulldozer.

.. list-table::
    :widths: 50 50
    :header-rows: 1
    
    * - FP32 Bulldozer 
      - FP16 Bulldozer

    * - .. image:: ../../../src/assignments/assignment_06/results/torch_32.png
           :width: 100%

      - .. image:: ../../../src/assignments/assignment_06/results/torch_16.png
           :width: 100%

Both results show the bulldozer clearly.
There are only minimal differences in the sharpness of the pictures.
Therefore, we will use the ``torch.float16`` tensors for the following tasks.

Task 2: Generating a Basic Config
---------------------------------

.. code-block:: bash
    :caption: Execution from machine-learning-compilers directory

    python3 -m src.assignments.assignment_06.main

a) Generate an initial config
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To use the ``generate_config`` function we first import the respective function from the :ref:`config <config>` file from assignment 5. 
We provide the ``generate_config`` function the entire einsum string ``acspx,bspy->abcyx`` and take the shape of the ``FP16`` input tensors.

.. code-block:: python 

    cfg = generate_config("acspx,bspy->abcyx", [tensor_acspx_fp16.shape, tensor_bspy_fp16.shape])

b) Resulting config
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The resulting config for the einsum string looks as follows:

.. code-block:: text

    Config(
      data_type  = DataType.FLOAT16
      prim_main  = PrimType.GEMM
      prim_last  = LastType.NONE
      prim_first = FirstType.ZERO
      dim_types  = ['M', 'M', 'K', 'K', 'M', 'N', 'N']
      exec_types = ['SEQ', 'SEQ', 'SEQ', 'SEQ', 'SEQ', 'SEQ', 'SEQ']
      dim_sizes  = [4, 3, 64, 64, 1536, 4, 1152]
      strides[0] = [18874368, 6291456, 98304, 1536, 1, 0, 0]
      strides[1] = [0, 0, 73728, 1152, 0, 4718592, 1]
      strides[2] = [21233664, 1769472, 0, 0, 1, 5308416, 1536]
    )
