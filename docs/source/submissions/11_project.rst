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