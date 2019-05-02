# Perform Bootstrapping and Multiple Imputation

The file *ensemble.java* is the main file to run.

The input data file name can be added on this line
*private static String str="iris";*

The path of the input arff file can be set from this line
*private static String path = "//media//Data//workspace//Missing//data//";*

The directory 'data' contains the common datasets used in our paper.

The missing ratio can be changed from this line
*int [] missingRatio= new int []{5,10,15,20,25,30};//Percentage of missing attribute values*
		
The output file will be generated with the name *inputfile-output.csv*

The package Weka-3.9.2 is required to run this program (use weka.jar, available at https://www.cs.waikato.ac.nz/ml/weka/downloading.html). This jar should be added externally if you are using an IDE (e.g. IntelliJ or Eclipse) or otherwise added in the path when compilng the code.

This code is tested on IntelliJ Idea 2019.1.1

If you use this software for your research, then please cite the following paper

**Khan, S.S., Ahmad, A. and Mihailidis, A., 2018. Bootstrapping and Multiple Imputation Ensemble Approaches for Missing Data. arXiv preprint arXiv:1802.00154.**
