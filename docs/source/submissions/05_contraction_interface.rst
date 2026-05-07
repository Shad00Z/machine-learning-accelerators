Contraction Interface and L2 Optimization
=========================================

Task 1: Config Class
--------------------

For our Config object we first defined enumeration types using `enum.Enum` for all of the configuration parameters.

.. literalinclude:: ../../../src/assignments/05_assignment/config.py
    :language: py
    :linenos:
    :lines: 9-44
    :caption: `Configuration parameter definitions`

After the definitions of the parameters we proceed with the `Config` class itself.
The `Config` class has 8 attributes: 

.. literalinclude:: ../../../src/assignments/05_assignment/config.py
    :language: py
    :linenos:
    :lines: 77-86
    :caption: `Config attributes`

For debugging purposes we also added a python `__repr__` function, which helps in preserving all the necessary information about the config object: 

.. literalinclude:: ../../../src/assignments/05_assignment/config.py
    :language: py
    :linenos:
    :lines: 88-102
    :caption: `__repr__ function`

Task 2: Generating a Basic Config
---------------------------------

After creating our basic setup with the configuration parameters and the `Config` itself, we can now create a `generate_config` function. 
This `generate_config` function takes an einsum string and a list of shapes for the input tensors and returns a basic `Config`. 

We broke the generation down to a four step procedure. 
The first step is to parse the einsum string to get output and input indices. 

.. literalinclude:: ../../../src/assignments/05_assignment/config.py
    :language: py
    :linenos:
    :lines: 190-196
    :caption: `Step 1: Parse the einsum string`

We make sure that the list of indices we retrieve from the einsum string matches with the input shapes that were handed to the `generate_config` function.

After the initial verification, we collect the dimension sizes and check for their consistency.

.. literalinclude:: ../../../src/assignments/05_assignment/config.py
    :language: py
    :linenos:
    :lines: 199-216
    :caption: `Step 2: Dimension index collection`

The collection is performed tensor by tensor. 
First, each tensor's indices and shapes are compared to rule out rank mismatches. 
Then, the single indices of the tensor are either added to the `dim_sizes` dictionary or an exception is raised, if an index already exists and it conflicts with the already existing dimension size.

With these `dim_sizes` we build the per-dimension attribute lists.

.. literalinclude:: ../../../src/assignments/05_assignment/config.py
    :language: py
    :linenos:
    :lines: 218-225
    :caption: `Step 3: Attribute list for dimensions`

To create an attribute list we iterate over our `dim_sizes` dictionary and classify each dimension. 

.. literalinclude:: ../../../src/assignments/05_assignment/config.py
    :language: py
    :linenos:
    :lines: 123-147
    :caption: `Dimension classification`

Dimension classification means we check in which tensors a dimension appears and according to that we classify the dimension.
For this third step, we also assign each dimension the `SEQ` execution type. 

In the fourth and final step we compute the strides for each tensor. 

.. literalinclude:: ../../../src/assignments/05_assignment/config.py
    :language: py
    :linenos:
    :lines: 228-232
    :caption: `Step 4: Stride computation`

First of all, we assign each dimension that is not appearing inside a tensor the value 0.
The calculation for the remaining dimensions is pretty straightforward. 

.. literalinclude:: ../../../src/assignments/05_assignment/config.py
    :language: py
    :linenos:
    :lines: 150-168
    :caption: `Stride multiplication`

We start with the rightmost index for a tensor and assign the value 1.
To calculate the strides we go from the rightmost to the leftmost index for each tensor.
Afterwards we multiply the size of the respective tensor (e.g. the rightmost) with the current stride and thereby receive the stride for next dimension index.

Task 3: Optimizer Class
-----------------------


