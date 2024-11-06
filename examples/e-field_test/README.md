Developed by Kit Joll - November 2024

This directory contains an example of how to run e-field MD using APTNN and cp2k.

In cp2k, there is an i-pi server, which the i-pi-driver.py file will connect to.
Then cp2k will send positions and request e-field induced forces from the APTNN model.
The APTNN model will then calculate the forces and send them back to cp2k.

This happens in a mixed-force environment. The (two) force evaluations happen in parallel, and the forces are mixed together before propagating the positions.

In this example we are running on a GPU cluster, so the APTNN model is running on a GPU and CPUs are used for cp2k. 

We request 25 cores and 1 gpu for this example.
8 cores are used for mpi APTNN (and gpu used by automatic detection).
The remaining 17 cores are used for cp2k.
These 17 cores are split by cp2k into 2 groups - one for the i-pi force evaluation section and 1 for nnp force evaluation.
We assign 16 (square power of 2) to NNP and 1 core to i-pi then the server communication.

Please see the run_files dir for the input files and the run.sh file for the submission script.
Please see the test_res dir for the expected output files.

