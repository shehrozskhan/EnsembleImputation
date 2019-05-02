# Perform Bootstrapping and Multiple Imputation

The file *ensemble.java* is the main file to run.

The input data file name can be added on this line

*private static String str="iris";*

The path of the input arff file can be set from this line

*private static String path = "//media//Data//workspace//Missing//data//";*

The output file will be generated with the name *inputfile-output.csv*

required package Weka-3.9.2 --> weka.jar (https://www.cs.waikato.ac.nz/ml/weka/downloading.html). This jar should be added as a if you are using an IDE (e.g. IntelliJ or Eclipse) or otherwise added when compilng the code.

This code is tested on IntelliJ Idea 2019.1.1

If you use this software for your research, then cite the following paper
Khan, S.S., Ahmad, A. and Mihailidis, A., 2018. Bootstrapping and Multiple Imputation Ensemble Approaches for Missing Data. arXiv preprint arXiv:1802.00154.
