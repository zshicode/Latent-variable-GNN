Installation
------------


Type

    ./run_GNN.sh DATA FOLD GPU
to run on dataset using fold number (1-10).

You can run

    ./run_GNN.sh DD 0 0
to run on DD dataset with 10-fold cross
validation on GPU #0.


Code
----

The detail implementation of Graph U-Net is in src/utils/ops.py.


Datasets
--------

Check the "data/README.md" for the format. 