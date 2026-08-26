
## Motivation:
This repo was my practice of lower level pytorch just to revise the dimensionality and optimizations that we need to take care while moving from conceptual to practical implementation. Like using Batched tensor operations for MHA instead of a for loop, accumulating training loss and only calculating it once per epoch after detaching it from the computation graph, learned more lower profiling of DL models like Cuda synch, profile warmups, chrome trace and compute vs overhead bound. This was mainly done so that I can revise the practical pytorch which I didn't touch in over a year and also get back into habit of reading papers. 

....might end up using this boiler plate along with some argparser arguments.......
