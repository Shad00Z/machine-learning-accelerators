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

    * - .. image:: ../../../src/assignments/06_assignment/results/torch_32.png
           :width: 100%

      - .. image:: ../../../src/assignments/06_assignment/results/torch_16.png
           :width: 100%

Both results show the bulldozer clearly.
There are only minimal differences in the sharpness of the pictures.
Therefore, we will use the ``torch.float16`` tensors for the following tasks.


