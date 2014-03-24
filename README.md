PageRank with MulticoreBSP
==========================

This repository contains an implementation of Google's PageRank algoritm,
implemented using multithreading with the [MulticoreBSP] library.

Implementations
---------------

Two different implementations of the algorithm are available. One calculates
the Google matrix and stores it in the memory as a dense matrix. By doing
this, the program can access all of the values very rapidly and make faster
calculations. The downside of this implementation is that it has a memory
complexity of O(N²). This means that it uses massive amounts of memory for
large link matrices.

A second implementation uses sorted linked lists to represents a sparse
matrix. Due to the nature of the adjustments made by Google to the hyperlink
matrix, it is possible to calculate the Google matrix values using sparse
data during execution very rapidly. Performance compared to the dense matrix
implementation is lower, but the difference will only show for very large
datasets, which would be nearly impossible to process with the dense matrix
implementation due to the memory requirements.

[MulticoreBSP]:http://www.multicorebsp.com/