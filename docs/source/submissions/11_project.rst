Project: cuTile for Local Diffusion
===================================

For the last three weeks of the machine-learning-accelerators course, we ought to select a personal project.
We have decided to optimize the denoising loop of a local text-to-image diffusion model.

As our model of choice we have selected the ``SD-Turbo`` text-to-image model.
The architecture of ``SD-Turbo`` is a U-Net.

Each denoising step runs the whole U-Net, which, for the ``SD-Turbo`` is built from 22 ResnetBlocks (and 16 transformer blocks).
For our selected model, the operations within such a block are:

- GroupNorm,
- SiLU,
- Conv2d, and
- Linear.

Further, on the transformer side there are:

- LayerNorm,
- GELU / GEGLU FFN, and
- the attention mechanism.

Looking at these operations it becomes clear that there are a lot of memory transfers happening.
Therefore, we plan to decrease the text-to-image time by fusing together several of these operations.
That means we are optimizing the diffusion model in regards to memory bandwidth.

Milestones
----------

To follow a clear plan for these three weeks we came up with five milestones.

0. Inspect SD-Turbo model in regards to memory requirements and realization on the DGX Spark GPU (Roofline model).
1. Reason about fusing candidates (layers) and tiling approaches.
2. Implement fusion kernel for two candidates (+ verify correctness via ``torch.allclose`` per block and benchmark).
3. Include the optimized kernel into the SD-Turbo model (+ verify end-to-end image equivalence).
4. Optimize the first candidate.
5. Repeat for the next candidate (Step 2).
6. Create CLI, Gradio UI to use the optimized model on the DGX Spark.

In regards to benchmarking we plan to:

- compare against ``torch.compile``,
- prove the approach with Nsight (bytes moved per step at each memory tier)
- keep track on the gain of each of our fusing approaches, and
- compare different tiling sizes and approaches.


Milestone 0: SD-Turbo Model
---------------------------

Additional requirements that are needed to generate images with the ``SD-Turbo`` model.

.. code-block:: bash
    :caption: requirements

    pip install diffusers
    pip install transformers
    # Silence warnings
    pip install accelerate
    pip install torchvision

Milestone 1: Candidates + Tiling
--------------------------------

After installing the required libraries, we then inspected a resnet and a transformer block of the model.

Resnet Block
^^^^^^^^^^^^^^

.. literalinclude:: ../_static/resnet.txt
    :language: text
    :caption: ResnetBlock2D

Based on this resnet-block it is clear that each operation (norm, SiLU) reads and writes :math:`2 \cdot 320 \cdot 64 \cdot 64 = 2.5MiB`.
Therefore, the clear candidates for a fusion are ``norm1 + SiLU`` and ``norm2 + SiLU``.
To achieve the highest performance for the resnet block, it is also possible to additionally fuse the denoising (``temb``) and the residual tail.

.. image:: ../_static/roofline.svg

The roofline model shows that the norm / activation operations are memory-bound, so fusing ``norm + SiLU`` makes sense.

Transformer Block
^^^^^^^^^^^^^^^^^

.. literalinclude:: ../_static/transformer.txt
    :language: text
    :caption: Transformer2DModel

For the transformer block, the ``LayerNorm + GEGLU`` are worth fusing with the matrix multiplications.

- ``LayerNorm -> mm1 (320, 2560) -> GEGLU -> mm2 (1280, 320)``

By fusing these operations, we plan to remove the separated memory passes for ``LayerNorm + GEGLU``.

Fusing
^^^^^^

After performing some initial tests, we verified that ``torch.compile`` already fuses ``norm + SiLU`` into a single kernel.
Therefore, this kernel serves as a warm-up kernel.

On the other hand, the ``Layernorm`` and ``GEGLU`` are executed as two separate kernels.
Therefore, fusing them with the matmul provides higher value compared to the ``torch.compile`` reference.

Tiling
^^^^^^

For the optimal tiling approach we have several constraints that have to be met, either from cuTile or the GPU.

From the cuTile side of things, we have to consider that the tile dimensions must be powers of two.

The Nvidia Spark GPU has ``24 MiB`` of L2 cache and ``48 SMs``.
The per-block activation (``2.5 MiB``) fits comfortably in L2, so at batch 1 it stays cache-resident.

For the ``norm + SiLU`` fusion we are looking at 32 groups and 320 channels.
Therefore, we can use 10 channels per group and thereby, have :math:`10 \times 64 \times 64 = 40,960` elements.

For the fusion of the matmul with the ``LayerNorm + GEGLU``, we keep the contraction pattern of the two linear layers with ``(320 -> 2560, 1280 -> 320)``.
We are going to fuse the ``LayerNorm`` as a first touch and then apply the ``GELU`` as a last touch for the first matmul.


Milestone 2: Fused GroupNorm + SiLU Kernel
------------------------------------------

GroupNorm Shape Coverage
^^^^^^^^^^^^^^^^^^^^^^^^^

A forward-pass survey of all 44 GroupNorm calls across the U-Net's ``ResnetBlock2D`` blocks reveals 14 distinct ``(channels, H, W, groups)`` shapes, of which the fused cuTile kernel currently serves only the ``(320, 64, 64, 32)`` case (16 % of calls).
Because the kernel is already shape-parametric -- the channel count is only a loop bound and every spatial extent is a power of two -- all 44 calls are kernel-eligible, so full coverage requires merely relaxing the ``GnSiluFused._is_supported`` gate rather than writing new kernels.

::

     chan    H    W  grps   Cg  count   now eligible
    ----------------------------------------------------
     1280    8    8    32   40     11            YES
      320   64   64    32   10      7   YES      YES
      640   32   32    32   20      6            YES
     1280   16   16    32   40      6            YES
     2560    8    8    32   80      3            YES
      640   64   64    32   20      2            YES
     2560   16   16    32   80      2            YES
      320   32   32    32   10      1            YES
      640   16   16    32   20      1            YES
      960   32   32    32   30      1            YES
      960   64   64    32   30      1            YES
     1280   32   32    32   40      1            YES
     1920   16   16    32   60      1            YES
     1920   32   32    32   60      1            YES
    ----------------------------------------------------
    total GN calls: 44   fused now: 7 (16%)   kernel-eligible: 44 (100%)
    distinct shapes: 14

After we applied the arbitrary shapes, we are now fusing everything:

::

     chan    H    W  grps   Cg  count   fusable
    --------------------------------------------
     1280    8    8    32   40     11       YES
      320   64   64    32   10      7       YES
      640   32   32    32   20      6       YES
     1280   16   16    32   40      6       YES
     2560    8    8    32   80      3       YES
      640   64   64    32   20      2       YES
     2560   16   16    32   80      2       YES
      320   32   32    32   10      1       YES
      640   16   16    32   20      1       YES
      960   32   32    32   30      1       YES
      960   64   64    32   30      1       YES
     1280   32   32    32   40      1       YES
     1920   16   16    32   60      1       YES
     1920   32   32    32   60      1       YES
    --------------------------------------------
    total GN calls: 44   fusable: 44 (100%)
    distinct shapes: 14


Test Execution
==============

.. code-block:: bash
    :caption: Test Execution

    python -m pytest test/diffusion/sd_turbo_fused/resnet/test_kernel_shapes.py -v