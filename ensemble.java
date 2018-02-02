package src;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

import weka.classifiers.trees.J48;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.experiment.Stats;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
import weka.filters.unsupervised.attribute.Add;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

public class ensemble {

	private static String str="iris";
	private static String path = "//home//shehroz//workspace//Missing//data//";	
	private static ReplaceMissingValues meifilter;
	private static EMImputation emfilter;
	private static BayesianMultipleImputation bmifilter;
	private static ArrayList<Filter> emfilterArray = new ArrayList<Filter> ();
	private static ArrayList<Filter> bmifilterArray = new ArrayList<Filter> ();


	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub

		String inputfile = path+str+".arff";
		String outputfile = path+str+"-output.csv";
		int Imputations = 5; //Number of Imputations
		int times=30; //Times to run the methods
		int ensembleSize=25; //Size of the Ensembles
		int [] missingRatio= new int []{5,10,15,20,25,30};//Percentage of missing attribute values
		double [] zvalue = {4,-4}; //To generate values for gaussian random imputation
		int folds = 2;
		String [] methodNames= {"No-Imp", "MEI", "RandI", "GRandI", "EM", "BMI", "BagNoImp", "BagMEI", "BagRandI", "BagGRandI",
				"BagEM", "BagBMI", "BagMIRandI", "BagMIGRandI", "BagMIEM", "BagMIBMI", "MIRandI", "MIGrandI", "MIEM", "MIBMI"};

		Random r = new Random();		
		//Generate seed for the number of times evaluation is run for a given missingness
		//??? should we need different seed per missingness???
		int [] seedTimes = new int [times];
		for (int i=0;i<times;i++) {
			seedTimes[i]=r.nextInt();
		}

		//Generate seed for different bootstrap samples
		int [] seedBag = new int [ensembleSize]; //To store random seeds for bagging, so all bagging data is same 
		for (int i=0;i<ensembleSize;i++) {
			seedBag[i]=r.nextInt();
		}

		//Evaluation of methods
		cvEvaluation(inputfile, outputfile, methodNames, seedTimes, folds, times, missingRatio, ensembleSize, Imputations, seedBag, zvalue); 
	} //End of main

	//Read Data
	public static Instances readData (String str) throws Exception {
		DataSource source = new DataSource(str);
		Instances data = source.getDataSet();
		// setting class attribute if the data format does not provide this information
		if (data.classIndex() == -1)
			data.setClassIndex(data.numAttributes() - 1);
		return data;
	}

	//Generate CV fold and do evaluation
	public static void cvEvaluation(String inputfile, String outputfile, String [] methodNames, int [] seedTimes, int folds,int times,
			int [] missingRatio, int ensembleSize, int Imputations, int [] seedBag, double [] zvalue) throws Exception {
		double [][] accuracy = new double [times][folds];
		double [][] accuracymei = new double [times][folds];
		double [][] accuracyavgrand = new double [times][folds];
		double [][] accuracyavgem = new double [times][folds];
		double [][] accuracyavgbmi = new double [times][folds];
		double [][] accuracyavggaussianrand = new double [times][folds];
		double [][] accuracyBag = new double [times][folds];
		double [][] accuracyBagmei = new double [times][folds];
		double [][] accuracyBagrand = new double [times][folds];
		double [][] accuracyBagGaussianrand = new double [times][folds];
		double [][] accuracyBagMIrand = new double [times][folds];
		double [][] accuracyBagMIGaussianrand = new double [times][folds];
		double [][] accuracyBagMIem = new double [times][folds];
		double [][] accuracyBagMIbmi = new double [times][folds];
		double [][] accuracyBagem = new double [times][folds];
		double [][] accuracyBagbmi = new double [times][folds];
		double [][] accuracyMIrand =  new double [times][folds];
		double [][] accuracyMIGaussianrand =  new double [times][folds];
		double [][] accuracyMIem =  new double [times][folds];
		double [][] accuracyMIbmi =  new double [times][folds];
		double [][] avgAccuracy = new double [missingRatio.length][methodNames.length]; 


		for (int m=0;m<missingRatio.length;m++) { //For every missing ratio
			System.out.println("\nmissingness="+missingRatio[m]+"%");
			for (int t=0;t<times;t++) { //For different times
				System.out.println("\ntimes="+(t+1));
				Instances data = readData(inputfile);				
				Instances missingData = introduceMissingnessColumn(data,missingRatio[m]);
				Random rand = new Random(seedTimes[t]);   // create seeded number generator
				Instances randData = new Instances(missingData);   // create copy of original data
				randData.randomize(rand);         // randomize data with number generator
				for (int n = 0; n < folds; n++) { //For every fold
					System.out.println("<<<<<Fold "+(n+1)+" >>>>>");

					//Create train and test set
					Instances train = randData.trainCV(folds, n);
					Instances test = randData.testCV(folds, n);	
					int lastAttribute = test.numAttributes()-1; 
					//boolean kappa =(missingRatio[m]==10 && t==0 && n==0);
					boolean kappa =(t==0 && n==0);
					/*** SINGLE IMPUTATION ***/
					//No Imputation, original data with missing value

					//Build Model on non-imputed data
					J48 rf=buildModel(train);
					//Test Model on non-imputed data and find accuracy
					accuracy[t][n]=testModel(test,lastAttribute,rf);

					//Mean Imputation
					Instances meiTrain = meanImputation(train);
					//Build Model for mean imputed values
					rf = buildModel(meiTrain);
					//Generate test set with mean imputation
					Instances meiTest=Filter.useFilter(test, meifilter);
					//Test Model on mean imputed data and find accuracy
					accuracymei[t][n]=testModel(meiTest,lastAttribute,rf);

					//Average Random Imputation
					Instances avgrandTrain = avgRandomImputation(train,Imputations);
					//Build model for random imputed values
					rf = buildModel(avgrandTrain);
					//Generate test set with random imputation
					Instances randTest=randomImputation(train,test); //Does it need to be average or just once?
					//Test Model on random imputed data and find accuracy
					accuracyavgrand[t][n] = testModel(randTest,lastAttribute,rf);

					//Average Gaussian Random Imputation
					Instances avggaussianrandTrain = avgGaussianRandomImputation(train,Imputations,zvalue);
					//Build model for random imputed values
					rf = buildModel(avggaussianrandTrain);
					//Generate test set with random imputation
					Instances grandTest=gaussianRandomImputation(train,test,zvalue); //Does it need to be average or just once?
					//Test Model on avg random imputed data and find accuracy
					accuracyavggaussianrand[t][n] = testModel(grandTest,lastAttribute,rf);

					//Average EM Imputation
					Instances avgemTrain = avgEMImputation(train, Imputations);
					//Build model for random imputed values
					rf=buildModel(avgemTrain);
					//Generate test sets with same parameters
					Instances avgemTest = avgEMImputationTest(test,Imputations);
					//Test Model on avg test set and find accuracy
					accuracyavgem[t][n] = testModel(avgemTest,lastAttribute,rf);

					/*
					//Average BM Imputation
					Instances avgbmiTrain = avgBMImputation(train, Imputations);
					//Build model for random imputed values
					rf=buildModel(avgbmiTrain);
					//Generate test sets with same parameters
					Instances avgbmiTest = avgBMImputationTest(test,Imputations);
					//Test Model on avg test set and find accuracy
					accuracyavgbmi[t][n] = testModel(avgbmiTest,lastAttribute,rf);
					 */

					/*** ENSEMBLE Methods ***/
					/** Bagging Based **/

					//No Imputation, original data with missing value//
					accuracyBag[t][n]=Bagging(train,test,ensembleSize,seedBag,lastAttribute,kappa,missingRatio[m]);

					//Single Imputation methods//
					//Mean Imputation
					accuracyBagmei[t][n]=BaggingMEI(train,test,ensembleSize,seedBag,lastAttribute,kappa,missingRatio[m]);
					//Random Imputation
					accuracyBagrand[t][n]=BaggingRand(train,test,ensembleSize,seedBag,lastAttribute,kappa,missingRatio[m]);
					//Gaussian Random Imputation
					accuracyBagGaussianrand[t][n]=BaggingGaussianRand(train,test,ensembleSize,seedBag,lastAttribute,zvalue,kappa,missingRatio[m]);
					//EM Imputation
					accuracyBagem[t][n]=BaggingEM(train,test,ensembleSize,seedBag,lastAttribute,kappa,missingRatio[m]);
					//BM Imputation
					//accuracyBagbmi[t][n]=BaggingBMI(train,test,ensembleSize,seedBag,lastAttribute,kappa,missingRatio[m]);

					//Multiple Imputation//
					//Random Imputation
					accuracyBagMIrand[t][n]=BaggingMIRand(train, test, ensembleSize, seedBag, Imputations, lastAttribute,kappa,missingRatio[m]);
					//Gaussian Random Imputation
					accuracyBagMIGaussianrand[t][n]=BaggingMIGaussianRand(train, test, ensembleSize, seedBag, Imputations, lastAttribute,zvalue,kappa,missingRatio[m]);
					//EM Imputation
					accuracyBagMIem[t][n]=BaggingMIEM(train, test, ensembleSize, seedBag, Imputations, lastAttribute,kappa,missingRatio[m]);
					//BM Imputation
					//accuracyBagMIbmi[t][n]=BaggingMIBMI(train, test, ensembleSize, seedBag, Imputations, lastAttribute,kappa,missingRatio[m]);

					/** Non-Bagging Based **/

					//Multiple Imputation ensembleSize times//
					//Random Imputation
					accuracyMIrand[t][n]=MIRand(train, test, ensembleSize, lastAttribute,kappa,missingRatio[m]);
					//Gaussian Random Imputation
					accuracyMIGaussianrand[t][n]=MIGaussianRand(train, test, ensembleSize, lastAttribute,zvalue,kappa,missingRatio[m]);
					//EM Imputation
					accuracyMIem[t][n]=MIEM(train, test, ensembleSize, lastAttribute,kappa,missingRatio[m]);
					//BM Imputation
					//accuracyMIbmi[t][n]=MIBMI(train, test, ensembleSize, lastAttribute,kappa,missingRatio[m]);
				} //end for folds

			} //end of times
			avgAccuracy[m][0]=averageAccuracy(accuracy,times,folds);
			avgAccuracy[m][1]=averageAccuracy(accuracymei,times,folds);
			avgAccuracy[m][2]=averageAccuracy(accuracyavgrand,times,folds);
			avgAccuracy[m][3]=averageAccuracy(accuracyavggaussianrand,times,folds);
			avgAccuracy[m][4]=averageAccuracy(accuracyavgem,times,folds);
			avgAccuracy[m][5]=averageAccuracy(accuracyavgbmi,times,folds);
			avgAccuracy[m][6]=averageAccuracy(accuracyBag,times,folds);
			avgAccuracy[m][7]=averageAccuracy(accuracyBagmei,times,folds);
			avgAccuracy[m][8]=averageAccuracy(accuracyBagrand,times,folds);
			avgAccuracy[m][9]=averageAccuracy(accuracyBagGaussianrand,times,folds);
			avgAccuracy[m][10]=averageAccuracy(accuracyBagem,times,folds);
			avgAccuracy[m][11]=averageAccuracy(accuracyBagbmi,times,folds);
			avgAccuracy[m][12]=averageAccuracy(accuracyBagMIrand,times,folds);
			avgAccuracy[m][13]=averageAccuracy(accuracyBagMIGaussianrand,times,folds);
			avgAccuracy[m][14]=averageAccuracy(accuracyBagMIem,times,folds);
			avgAccuracy[m][15]=averageAccuracy(accuracyBagMIbmi,times,folds);	
			avgAccuracy[m][16]=averageAccuracy(accuracyMIrand,times,folds);	
			avgAccuracy[m][17]=averageAccuracy(accuracyMIGaussianrand,times,folds);	
			avgAccuracy[m][18]=averageAccuracy(accuracyMIem,times,folds);	
			avgAccuracy[m][19]=averageAccuracy(accuracyMIbmi,times,folds);	
		} //end missingRatio
		//Print the output on screen and in text file
		printOutput(avgAccuracy,missingRatio,outputfile,methodNames);
	} //end of cvEvaluation

	//Introduce missingness
	public static Instances introduceMissingness (Instances data, int Missingness) {
		int totalValues = data.numInstances() * (data.numAttributes()-1);
		System.out.println("totalValues="+totalValues);
		int missingValues = (int) (Missingness * totalValues/100);
		System.out.println("missingValues="+missingValues);
		int[] temp = new int [missingValues];
		//Generate 'missingValues' amount of random number (non-repetetive)
		temp = randomNumber(missingValues, totalValues);
		for(int i=0;i<missingValues;i++) System.out.print(temp[i]+" ");
		System.out.println();
		//Replace missing values with '?'
		for(int i=0;i<missingValues;i++){
			int rem = (int) temp[i]%(data.numAttributes()-1);
			int quo = (int) temp[i]/(data.numAttributes()-1);
			data.instance(quo).setValue(rem, Double.NaN);
		}
		//System.out.println(data);
		return data;

	} //end for introduceMissingness

	//Introduce missingness - Row-wise
	public static Instances introduceMissingnessRow (Instances data, int Missingness) {
		//int totalValues = data.numInstances() * (data.numAttributes()-1);
		int totalValues = data.numAttributes()-1;
		System.out.println("totalValues="+totalValues);
		int missingValues = (int) (Missingness * totalValues/100);
		System.out.println("missingValues="+missingValues);
		for (int i=0;i<data.numInstances();i++) {
			int[] temp = new int [missingValues];
			//Generate 'missingValues' amount of random number (non-repetetive)
			temp = randomNumber(missingValues, totalValues);
			//Replace missing values with '?'
			for(int j=0;j<temp.length;j++){
				data.instance(i).setValue(temp[j], Double.NaN);
			}
		}
		//System.out.println(data);
		return data;
	} //end for introduceMissingness Row-wise

	//Introduce missingness - Column-wise
	public static Instances introduceMissingnessColumn (Instances data, int Missingness) {
		//int totalValues = data.numInstances() * (data.numAttributes()-1);
		int totalValues = data.numInstances();
		int [][] mat = new int [data.numInstances()][data.numAttributes()-1];
		int [] sum = new int [data.numInstances()];
		System.out.println("totalValues="+totalValues);
		int missingValues = (int) (Missingness * totalValues/100);
		System.out.println("missingValues="+missingValues);
		for (int i=0;i<data.numAttributes()-1;i++) {
			int[] temp = new int [missingValues];
			//Generate 'missingValues' amount of random number (non-repetitive)
			temp = randomNumber(missingValues, totalValues);
			for(int j=0;j<temp.length;j++){
				mat[temp[j]][i]=1; //set value 1 if missing value at this index
				sum[temp[j]]++; //sum the row
				//if it is the last attribute and all the values of a data object is missing
				if (i==data.numAttributes()-2 && sum[temp[j]]==(data.numAttributes()-1)) {
					//System.out.println("sum="+sum[temp[j]]+" j="+temp[j]);
					sum[temp[j]]--; //Leave this element from imputation
					mat[temp[j]][i]=0;//unset, do not impute this value
					//find the next available index from the dataset from top that can be imputed
					for(int k=0;k<data.numInstances();k++) {
						//if sum is less than num of attributes and this index has not been used before
						System.out.println("k="+k+" i="+i);
						if (sum[k]<(data.numAttributes()-2) && mat[k][i]==0) {
							System.out.println("index to impute="+k);
							data.instance(k).setValue(i, Double.NaN);
							mat[k][i]=1;
							sum[k]++;
							break;
						}
					} //end for k
					System.out.println();
				} else {
					//Replace missing values with '?'
					data.instance(temp[j]).setValue(i, Double.NaN);
				}
			} //end for j
			//System.out.println();
		}//end for i
		//System.out.println(data);
		return data;
	} //end for introduceMissingness Column-wise

	//Generate non-repetitive random numbers
	public static int[] randomNumber(int N, int range){
		Random r = new Random();
		int [] temp = new int [N];
		for(int i=0; i<N; i++) {
			int p = r.nextInt(range);
			temp[i]=p;
			for(int j=i-1;j>=0;j--) {
				if(p==temp[j]) {
					i--;
					break;
				}
			}
		}
		return temp;
	} //end of randomNumber()

	//Build Model
	public static J48 buildModel(Instances train) throws Exception {
		J48 rf = new J48();
		rf.buildClassifier(train);
		return rf;
	}

	//Test Model and compute accuracy
	public static double testModel(Instances test, int lastAttribute, J48 rf) throws Exception {
		int [][] cm = new int [test.numClasses()][test.numClasses()];
		for (int i=0;i<test.numInstances();i++) {
			int actual = (int) test.instance(i).value(lastAttribute);
			//System.out.println(test.instance(i).value(lastAttribute));
			int predict=(int) rf.classifyInstance(test.instance(i));
			cm[actual][predict]=cm[actual][predict]+1;
		}
		double sum=0;
		for (int i=0;i<test.numClasses();i++) {
			sum=sum+cm[i][i];
		}
		double accuracy=sum/test.numInstances();
		//System.out.println("accuracy="+accuracy);
		return accuracy;
	} //end for testModel

	//Test ensemble models and compute accuracy 
	public static double testModel(Instances test, int lastAttribute, int ensembleSize, ArrayList<ArrayList<String>> ensembleLabel) {
		int [][] cm = new int [test.numClasses()][test.numClasses()];
		for (int i=0;i<test.numInstances();i++) {
			int [] count = new int [test.numClasses()];
			for (int e=0;e<ensembleSize;e++) {
				String s = ensembleLabel.get(e).get(i);
				int L = Integer.valueOf(s).intValue();
				//System.out.print(L+" ");
				count[L]++;										
			}
			int predict = findLargestIndex(count);
			//System.out.println(" ==> "+predict);			
			int actual = (int) test.instance(i).value(lastAttribute);
			cm[actual][predict]=cm[actual][predict]+1;
		}
		double sum=0;
		for (int i=0;i<test.numClasses();i++) {
			sum=sum+cm[i][i];
		}
		double accuracy=sum/test.numInstances();
		//System.out.println("accuracy="+accuracy);
		return accuracy;
	} //end for testModel

	//Get class labels
	public static ArrayList<String> ensembleLabels(Instances test, J48 rf) throws Exception {
		ArrayList<String> temp = new ArrayList<String>();
		for (int i=0;i<test.numInstances();i++) {
			int label=(int) rf.classifyInstance(test.instance(i));
			temp.add(i, String.valueOf(label));				
		}
		return temp;
	}

	//Function to print ensemble labels outputs to a file for computing Kappa Statistics
	public static void kappaStat(Instances test, int lastAttribute, ArrayList<ArrayList<String>> Label, String funcname,
			int missRatio) throws IOException {
		String dirPath=path+str;
		File file = new File(path+str);
		if (!file.exists()) {
			if (file.mkdir()) {
				System.out.println("Directory "+str+" is created!");
			} else {
				System.out.println("Failed to create "+str+" directory!");
			}
		}

		String outputfn = dirPath+"//"+str+"-"+funcname+String.valueOf(missRatio)+".csv";
		BufferedWriter out = new BufferedWriter(new FileWriter(outputfn));
		//extract true class labels
		ArrayList<String> truelabels = new ArrayList<String>();
		for (int i=0;i<test.numInstances();i++) {
			int L= (int) test.instance(i).value(lastAttribute);
			truelabels.add(i, String.valueOf(L));
		}
		Label.add(truelabels);
		for (int j=0;j<test.numInstances();j++) {
			for (int i=0;i<Label.size();i++) {
				out.write(Label.get(i).get(j)+", ");
			}
			out.newLine();;
		}
		out.close();
	}//end for kappaStat

	//Ensemble Learning for non-imputed data
	public static double Bagging(Instances train, Instances test, int ensembleSize, int [] seedBag, int lastAttribute,boolean kappa,
			int missRatio) throws Exception {
		ArrayList<ArrayList<String>> Label = new ArrayList<ArrayList<String>>();
		for (int e=0;e<ensembleSize;e++) {
			Instances edata = bootstrapSamples(train,seedBag[e]);
			//Build Model on non-imputed data
			J48 rf=buildModel(edata);
			//Test model on non-imputed data and find class label
			ArrayList<String> tlabel = ensembleLabels(test,rf);
			Label.add(tlabel);						
		} //End of Ensemble
		double accuracy=testModel(test,lastAttribute,ensembleSize,Label);
		//Extract function name
		String funcname = new Object(){}.getClass().getEnclosingMethod().getName();
		//Print output file for kappa statistics
		if (kappa==true)
			kappaStat(test,lastAttribute,Label,funcname,missRatio);
		return accuracy;
	} //end for Bagging

	//Ensemble Learning for Mean imputed data
	public static double BaggingMEI(Instances train, Instances test, int ensembleSize, int [] seedBag, int lastAttribute, 
			boolean kappa, int missRatio) throws Exception {
		double accuracy;
		ArrayList<ArrayList<String>> Label = new ArrayList<ArrayList<String>>();
		Instances meiTest = new Instances (test, test.numInstances());
		for (int e=0;e<ensembleSize;e++) {
			Instances edata = bootstrapSamples(train,seedBag[e]);
			//Mean Imputation
			Instances meiTrain = meanImputation(edata);			
			//Build Model on mean-imputed data
			J48 rf=buildModel(meiTrain);
			//Generate test set with mean imputation
			meiTest= Filter.useFilter(test, meifilter);
			//Test model on mean-imputed data and find class label
			ArrayList<String> tlabel = ensembleLabels(meiTest,rf);
			Label.add(tlabel);						
		} //End of Ensemble
		accuracy=testModel(meiTest,lastAttribute,ensembleSize,Label);
		//Extract function name
		String funcname = new Object(){}.getClass().getEnclosingMethod().getName();
		//Print output file for kappa statistics
		if (kappa==true)
			kappaStat(test,lastAttribute,Label,funcname,missRatio);
		return accuracy;
	} //end for BaggingMEI

	//Ensemble Learning for Random imputed data
	public static double BaggingRand(Instances train, Instances test, int ensembleSize, int [] seedBag, int lastAttribute, 
			boolean kappa,int missRatio) throws Exception {
		ArrayList<ArrayList<String>> Label = new ArrayList<ArrayList<String>>();
		Instances randTest = new Instances (test, test.numInstances());
		for (int e=0;e<ensembleSize;e++) {
			Instances edata = bootstrapSamples(train,seedBag[e]);
			//Random Imputation
			Instances randTrain = randomImputation(edata,edata);			
			//Build Model on random-imputed data
			J48 rf=buildModel(randTrain);
			//Generate test set with random imputation
			randTest=randomImputation(edata,test);
			//Test model on random-imputed data and find class label
			ArrayList<String> tlabel = ensembleLabels(randTest,rf);
			Label.add(tlabel);						
		} //End of Ensemble
		double accuracy=testModel(randTest,lastAttribute,ensembleSize,Label);
		//Extract function name
		String funcname = new Object(){}.getClass().getEnclosingMethod().getName();
		//Print output file for kappa statistics
		if (kappa==true)
			kappaStat(test,lastAttribute,Label,funcname,missRatio);
		return accuracy;
	} //end for BaggingRand


	//Ensemble Learning for Gaussian Random imputed data
	public static double BaggingGaussianRand(Instances train, Instances test, int ensembleSize, int [] seedBag, int lastAttribute,
			double [] zvalue, boolean kappa,int missRatio) throws Exception {
		ArrayList<ArrayList<String>> Label = new ArrayList<ArrayList<String>>();
		Instances randTest = new Instances (test, test.numInstances());
		for (int e=0;e<ensembleSize;e++) {
			Instances edata = bootstrapSamples(train,seedBag[e]);
			//Random Imputation
			Instances randTrain = gaussianRandomImputation(edata,edata,zvalue);			
			//Build Model on random-imputed data
			J48 rf=buildModel(randTrain);
			//Generate test set with random imputation
			randTest=gaussianRandomImputation(edata,test,zvalue);
			//Test model on random-imputed data and find class label
			ArrayList<String> tlabel = ensembleLabels(randTest,rf);
			Label.add(tlabel);						
		} //End of Ensemble
		double accuracy=testModel(randTest,lastAttribute,ensembleSize,Label);
		//Extract function name
		String funcname = new Object(){}.getClass().getEnclosingMethod().getName();
		//Print output file for kappa statistics
		if (kappa==true)
			kappaStat(test,lastAttribute,Label,funcname,missRatio);
		return accuracy;
	} //end for BaggingGaussianRand


	//Ensemble Learning for EM imputed data
	public static double BaggingEM(Instances train, Instances test, int ensembleSize, int [] seedBag, int lastAttribute, 
			boolean kappa,int missRatio) throws Exception {
		ArrayList<ArrayList<String>> Label = new ArrayList<ArrayList<String>>();
		Instances emTest = new Instances (test, test.numInstances());
		for (int e=0;e<ensembleSize;e++) {
			Instances edata = bootstrapSamples(train,seedBag[e]);
			//Random Imputation
			Instances emTrain = EMaxImputation(edata);			
			//Build Model on random-imputed data
			J48 rf=buildModel(emTrain);
			//Generate test set with random imputation
			emTest = Filter.useFilter(test, emfilter);
			//Test model on random-imputed data and find class label
			ArrayList<String> tlabel = ensembleLabels(emTest,rf);
			Label.add(tlabel);						
		} //End of Ensemble
		double accuracy=testModel(emTest,lastAttribute,ensembleSize,Label);
		//Extract function name
		String funcname = new Object(){}.getClass().getEnclosingMethod().getName();
		//Print output file for kappa statistics
		if (kappa==true)
			kappaStat(test,lastAttribute,Label,funcname,missRatio);
		return accuracy;
	} //end for BaggingEM

	//Ensemble Learning for EM imputed data
	public static double BaggingBMI(Instances train, Instances test, int ensembleSize, int [] seedBag, int lastAttribute, 
			boolean kappa,int missRatio) throws Exception {
		ArrayList<ArrayList<String>> Label = new ArrayList<ArrayList<String>>();
		Instances bmiTest = new Instances (test, test.numInstances());
		for (int e=0;e<ensembleSize;e++) {
			Instances edata = bootstrapSamples(train,seedBag[e]);
			//Random Imputation
			Instances bmiTrain = BMImputation(edata);			
			//Build Model on random-imputed data
			J48 rf=buildModel(bmiTrain);
			//Generate test set with random imputation
			bmiTest = Filter.useFilter(test, bmifilter);
			//Test model on random-imputed data and find class label
			ArrayList<String> tlabel = ensembleLabels(bmiTest,rf);
			Label.add(tlabel);						
		} //End of Ensemble
		double accuracy=testModel(bmiTest,lastAttribute,ensembleSize,Label);
		//Extract function name
		String funcname = new Object(){}.getClass().getEnclosingMethod().getName();
		//Print output file for kappa statistics
		if (kappa==true)
			kappaStat(test,lastAttribute,Label,funcname,missRatio);

		return accuracy;
	} //end for BaggingBMI

	//Bagging followed by multiple imputation for Random imputed data
	public static double BaggingMIRand(Instances train, Instances test, int ensembleSize, int [] seedBag, int Imputations, 
			int lastAttribute, boolean kappa,int missRatio) throws Exception {
		ArrayList<ArrayList<String>> Label = new ArrayList<ArrayList<String>>();
		if (ensembleSize%Imputations != 0)
			throw new Exception("Size of Ensemble should be divisible by Number of Imputations ");
		for (int i=0;i<ensembleSize/Imputations;i++) {
			//Create data set after resampling
			Instances edata = bootstrapSamples(train,seedBag[i]);
			for (int j=0;j<Imputations;j++) {
				//Random Imputation
				Instances rdata = randomImputation(edata, edata);
				//Build Model on random imputed data
				J48 rf = buildModel(rdata);
				//Generate test set with random imputation
				Instances tdata = randomImputation(edata,test);
				//Test model on random-imputed data and find class label
				ArrayList<String> tlabel = ensembleLabels(tdata,rf);
				Label.add(tlabel);						
			} //end for Imputations			
		} //end for ensembleSize/Imputations
		double accuracy = testModel(test,lastAttribute,ensembleSize,Label);
		//Extract function name
		String funcname = new Object(){}.getClass().getEnclosingMethod().getName();
		//Print output file for kappa statistics
		if (kappa==true)
			kappaStat(test,lastAttribute,Label,funcname,missRatio);
		return accuracy;
	} //end for BaggingMIRand

	//Bagging followed by multiple imputation for Gaussian Random imputed data
	public static double BaggingMIGaussianRand(Instances train, Instances test, int ensembleSize, int [] seedBag, int Imputations, 
			int lastAttribute, double [] zvalue, boolean kappa,int missRatio) throws Exception {
		ArrayList<ArrayList<String>> Label = new ArrayList<ArrayList<String>>();
		if (ensembleSize%Imputations != 0)
			throw new Exception("Size of Ensemble should be divisible by Number of Imputations ");
		for (int i=0;i<ensembleSize/Imputations;i++) {
			//Create data set after resampling
			Instances edata = bootstrapSamples(train,seedBag[i]);
			for (int j=0;j<Imputations;j++) {
				//Random Imputation
				Instances rdata = gaussianRandomImputation(edata, edata,zvalue);
				//Build Model on random imputed data
				J48 rf = buildModel(rdata);
				//Generate test set with random imputation
				Instances tdata = gaussianRandomImputation(edata,test,zvalue);
				//Test model on random-imputed data and find class label
				ArrayList<String> tlabel = ensembleLabels(tdata,rf);
				Label.add(tlabel);						
			} //end for Imputations			
		} //end for ensembleSize/Imputations
		double accuracy = testModel(test,lastAttribute,ensembleSize,Label);
		//Extract function name
		String funcname = new Object(){}.getClass().getEnclosingMethod().getName();
		//Print output file for kappa statistics
		if (kappa==true)
			kappaStat(test,lastAttribute,Label,funcname,missRatio);
		return accuracy;
	} //end for BaggingMIGaussianRand

	//Bagging followed by multiple imputation for EM imputed data
	public static double BaggingMIEM(Instances train, Instances test, int ensembleSize, int [] seedBag, int Imputations, 
			int lastAttribute, boolean kappa,int missRatio) throws Exception {
		ArrayList<ArrayList<String>> Label = new ArrayList<ArrayList<String>>();
		if (ensembleSize%Imputations != 0)
			throw new Exception("Size of Ensemble should be divisible by Number of Imputations ");
		for (int i=0;i<ensembleSize/Imputations;i++) {
			//Create data set after resampling
			Instances edata = bootstrapSamples(train,seedBag[i]);
			for (int j=0;j<Imputations;j++) {
				//EM Imputation
				Instances rdata = EMaxImputation(edata);
				//Build Model on EM imputed data
				J48 rf = buildModel(rdata);
				//Generate test set with EM imputation
				Instances tdata = Filter.useFilter(test, emfilter);
				//Test model on EM-imputed data and find class label
				ArrayList<String> tlabel = ensembleLabels(tdata,rf);
				Label.add(tlabel);						
			} //end for Imputations			
		} //end for ensembleSize/Imputations
		double accuracy = testModel(test,lastAttribute,ensembleSize,Label);
		//Extract function name
		String funcname = new Object(){}.getClass().getEnclosingMethod().getName();
		//Print output file for kappa statistics
		if (kappa==true)
			kappaStat(test,lastAttribute,Label,funcname,missRatio);
		return accuracy;
	} //end for BaggingMIEM

	//Bagging followed by multiple imputation for BM imputed data
	public static double BaggingMIBMI(Instances train, Instances test, int ensembleSize, int [] seedBag, int Imputations, 
			int lastAttribute, boolean kappa,int missRatio) throws Exception {
		ArrayList<ArrayList<String>> Label = new ArrayList<ArrayList<String>>();
		if (ensembleSize%Imputations != 0)
			throw new Exception("Size of Ensemble should be divisible by Number of Imputations ");
		for (int i=0;i<ensembleSize/Imputations;i++) {
			//Create data set after resampling
			Instances edata = bootstrapSamples(train,seedBag[i]);
			for (int j=0;j<Imputations;j++) {
				//BM Imputation
				Instances rdata = BMImputation(edata);
				//Build Model on EM imputed data
				J48 rf = buildModel(rdata);
				//Generate test set with BM imputation
				Instances tdata = Filter.useFilter(test, bmifilter);
				//Test model on BM-imputed data and find class label
				ArrayList<String> tlabel = ensembleLabels(tdata,rf);
				Label.add(tlabel);						
			} //end for Imputations			
		} //end for ensembleSize/Imputations
		double accuracy = testModel(test,lastAttribute,ensembleSize,Label);
		//Extract function name
		String funcname = new Object(){}.getClass().getEnclosingMethod().getName();
		//Print output file for kappa statistics
		if (kappa==true)
			kappaStat(test,lastAttribute,Label,funcname,missRatio);
		return accuracy;
	} //end for BaggingMIBMI

	//Multiple Random Imputations 'ensembleSize' times on the same data
	public static double MIRand(Instances train, Instances test, int ensembleSize, int lastAttribute, boolean kappa,int missRatio) throws Exception {
		ArrayList<ArrayList<String>> Label = new ArrayList<ArrayList<String>>();
		for (int i=0;i<ensembleSize;i++) {
			Instances randTrain = randomImputation(train, train);
			//Build model for random imputed values
			J48 rf = buildModel(randTrain);
			//Generate test set with random imputation
			Instances randTest=randomImputation(train,test);
			//Test model on random-imputed data and find class label
			ArrayList<String> tlabel = ensembleLabels(randTest,rf);
			Label.add(tlabel);									
		}
		double accuracy=testModel(test,lastAttribute,ensembleSize,Label);
		//Extract function name
		String funcname = new Object(){}.getClass().getEnclosingMethod().getName();
		//Print output file for kappa statistics
		if (kappa==true)
			kappaStat(test,lastAttribute,Label,funcname,missRatio);
		return accuracy;
	}

	//Multiple Gaussian Random Imputations 'ensembleSize' times on the same data
	public static double MIGaussianRand(Instances train, Instances test, int ensembleSize, int lastAttribute, double [] zvalue, 
			boolean kappa,int missRatio) throws Exception {
		ArrayList<ArrayList<String>> Label = new ArrayList<ArrayList<String>>();
		for (int i=0;i<ensembleSize;i++) {
			Instances randTrain = gaussianRandomImputation(train, train, zvalue);
			//Build model for random imputed values
			J48 rf = buildModel(randTrain);
			//Generate test set with random imputation
			Instances randTest=gaussianRandomImputation(train,test, zvalue);
			//Test model on random-imputed data and find class label
			ArrayList<String> tlabel = ensembleLabels(randTest,rf);
			Label.add(tlabel);									
		}
		double accuracy=testModel(test,lastAttribute,ensembleSize,Label);
		//Extract function name
		String funcname = new Object(){}.getClass().getEnclosingMethod().getName();
		//Print output file for kappa statistics
		if (kappa==true)
			kappaStat(test,lastAttribute,Label,funcname,missRatio);
		return accuracy;
	}

	//Multiple EM Imputations 'ensembleSize' times on the same data
	public static double MIEM(Instances train, Instances test, int ensembleSize, int lastAttribute, boolean kappa,int missRatio) throws Exception {
		ArrayList<ArrayList<String>> Label = new ArrayList<ArrayList<String>>();
		for (int i=0;i<ensembleSize;i++) {
			Instances randTrain = EMaxImputation(train);
			//Build model for random imputed values
			J48 rf = buildModel(randTrain);
			//Generate test set with random imputation
			Instances randTest=Filter.useFilter(test, emfilter);
			//Test model on random-imputed data and find class label
			ArrayList<String> tlabel = ensembleLabels(randTest,rf);
			Label.add(tlabel);									
		}
		double accuracy=testModel(test,lastAttribute,ensembleSize,Label);
		//Extract function name
		String funcname = new Object(){}.getClass().getEnclosingMethod().getName();
		//Print output file for kappa statistics
		if (kappa==true)
			kappaStat(test,lastAttribute,Label,funcname,missRatio);
		return accuracy;
	}

	//Multiple BM Imputations 'ensembleSize' times on the same data
	public static double MIBMI(Instances train, Instances test, int ensembleSize, int lastAttribute, boolean kappa,int missRatio) throws Exception {
		ArrayList<ArrayList<String>> Label = new ArrayList<ArrayList<String>>();
		for (int i=0;i<ensembleSize;i++) {
			Instances randTrain = BMImputation(train);
			//Build model for random imputed values
			J48 rf = buildModel(randTrain);
			//Generate test set with random imputation
			Instances randTest=Filter.useFilter(test, bmifilter);
			//Test model on random-imputed data and find class label
			ArrayList<String> tlabel = ensembleLabels(randTest,rf);
			Label.add(tlabel);									
		}
		double accuracy=testModel(test,lastAttribute,ensembleSize,Label);
		//Extract function name
		String funcname = new Object(){}.getClass().getEnclosingMethod().getName();
		//Print output file for kappa statistics
		if (kappa==true)
			kappaStat(test,lastAttribute,Label,funcname,missRatio);
		return accuracy;
	}

	//find average accuracy across 'times' and 'folds'
	public static double averageAccuracy(double [][]accuracy,int times, int folds){
		double sum=0;
		for (int i=0;i<times;i++) {
			for (int j=0;j<folds;j++) {
				sum+=accuracy[i][j];
			}
		}
		double avg = sum/(times*folds);
		//System.out.println("avgAcc="+avg);
		return avg;
	} //end for averageAccuracy


	//Find the index of largest element in an array
	public static int findLargestIndex(int [] array) {
		int largest = array[0], index = 0;
		for (int i = 1; i < array.length; i++) {
			if ( array[i] > largest ) {
				largest = array[i];
				index = i;
			}
		}
		return index;
	}
	//Mean Imputation
	public static Instances meanImputation (Instances data) throws Exception {
		//Mean Imputation
		Instances mei = new Instances (data, data.numInstances());
		meifilter = new ReplaceMissingValues();
		meifilter.setInputFormat(data);
		mei = Filter.useFilter(data, meifilter);
		return mei;
	}

	//EM Imputation on trainset
	public static Instances EMaxImputation (Instances data) throws Exception {
		//EMImputation - changes the order of data
		Instances emi = new Instances (data, data.numInstances());
		emfilter = new EMImputation();
		emfilter.setUseRidgePrior(true);
		emfilter.setRidge(0.01);
		emfilter.setInputFormat(data);
		emi = Filter.useFilter(data, emfilter);
		//System.out.println(emi);
		return emi;
	}

	//EM Imputation on testset
	public static Instances EMaxImputationTest (Instances data, int index) throws Exception {
		Instances emi = new Instances (data, data.numInstances());
		emfilterArray.get(index).setInputFormat(data);
		emi = Filter.useFilter(data, emfilterArray.get(index));

		return emi;
	}

	//Prepare nominal class labels string (to be added later on)
	public static String extractClassString(Instances data) {
		StringBuilder sb = new StringBuilder();
		String classStr = new String();
		for (int i=0;i<data.classAttribute().numValues();i++) {
			sb=sb.append(data.classAttribute().value(i).toString()+",");
		}		
		int dex=sb.length()-1;
		sb.deleteCharAt(dex);//Remove the last comma
		classStr=sb.toString();
		return classStr;
	}

	//Hack 2 - Add this newlabels as the new class label, first remove the existing numeric attribute
	public static Instances hack2(Instances emi, String classStr, String [] newlabels) throws Exception {
		Remove re = new Remove();
		String str1 ="last";
		re.setAttributeIndices(str1);
		re.setInputFormat(emi);
		emi=Filter.useFilter(emi, re);

		Add add = new Add();
		add.setAttributeIndex(str1);
		add.setNominalLabels(classStr);
		add.setAttributeName("class");
		add.setInputFormat(emi);
		emi=Filter.useFilter(emi, add);
		for (int i=0;i<emi.numInstances();i++) {
			emi.instance(i).setValue(emi.numAttributes()-1, newlabels[i]);
		}
		return emi;
	}

	//Hack1 to add a new class attribute because original order is disturbed by EM
	public static Instances hack1(Instances data) throws Exception {		
		//Remove class attribute
		Remove re = new Remove();
		String str1 ="last";
		re.setAttributeIndices(str1);
		re.setInputFormat(data);
		data = Filter.useFilter(data, re);

		//Add new class attribute that corresponds to instance number. The numeric class label for instance will not be impute
		// and can be used to get back the original labels
		Add add = new Add();
		add.setAttributeIndex(str1);
		add.setAttributeName("NewNumeric");
		add.setInputFormat(data);
		data = Filter.useFilter(data, add);
		for(int i=0;i<data.numInstances();i++) {
			data.instance(i).setValue(data.numAttributes()-1, i);
		}
		return data;
	}

	//Bayesian Multiple Imputation on trainset
	public static Instances BMImputation (Instances data) throws Exception {
		Instances bmi = new Instances (data, data.numInstances());
		bmifilter = new BayesianMultipleImputation();
		bmifilter.setNumImputations(1);
		bmifilter.setUseRidgePrior(true);
		bmifilter.setRidge(0.01);
		bmifilter.setInputFormat(data);
		bmi = Filter.useFilter(data, bmifilter);
		return bmi;
	}

	//Bayesian Multiple Imputation on testset
	public static Instances BMImputationTest (Instances data, int index) throws Exception {
		//EMImputation - changes the order of data
		Instances emi = new Instances (data, data.numInstances());
		bmifilterArray.get(index).setInputFormat(data);
		emi = Filter.useFilter(data, bmifilterArray.get(index));
		//System.out.println(emi);
		return emi;
	}

	//Average Random Imputation
	public static Instances avgRandomImputation (Instances train, int Imputations) {
		double [][] val = new double[train.numInstances()][train.numAttributes()-1];
		//find the average of all the imputations
		for (int k=0;k<Imputations;k++) {
			Instances ri = randomImputation(train,train);
			for (int i=0;i<ri.numInstances();i++) {
				for (int j=0;j<ri.numAttributes()-1;j++) {
					val[i][j] += ri.instance(i).value(j)/Imputations;
				}
			}
		}

		//Extract the class labels in an array
		String [] trainClasslabel = extractClassLabels(train);

		//Construct a new data that contains average of random imputed values
		Instances avgdata = avgData(train, trainClasslabel, val);

		return avgdata;		
	}

	//Average Gaussian Random Imputation
	public static Instances avgGaussianRandomImputation (Instances train, int Imputations, double [] zvalue) {
		double [][] val = new double[train.numInstances()][train.numAttributes()-1];
		//find the average of all the imputations
		for (int k=0;k<Imputations;k++) {
			Instances ri = gaussianRandomImputation(train,train, zvalue);
			for (int i=0;i<ri.numInstances();i++) {
				for (int j=0;j<ri.numAttributes()-1;j++) {
					val[i][j] += ri.instance(i).value(j)/Imputations;
				}
			}
		}

		//Extract the class labels in an array
		String [] trainClasslabel = extractClassLabels(train);

		//Construct a new data that contains average of random imputed values
		Instances avgdata = avgData(train, trainClasslabel, val);

		return avgdata;		
	}

	//Average EM imputation on train set
	public static Instances avgEMImputation (Instances train, int Imputations) throws Exception {
		double [][] val = new double[train.numInstances()][train.numAttributes()-1];
		//find the average of all the imputations
		Instances ri = new Instances(train, train.numInstances());
		for (int k=0;k<Imputations;k++) {
			ri = EMaxImputation(train);
			//Check
			if (ri.numInstances()!=train.numInstances()) {
				System.out.println(ri.numInstances());
				throw new Exception("size mismatch in Trainset Avg. EMImputation");
			}
			emfilterArray.add(k, emfilter);
			for (int i=0;i<train.numInstances();i++) {
				for (int j=0;j<train.numAttributes()-1;j++) {
					val[i][j] += ri.instance(i).value(j)/Imputations;
				}
			}
		}

		//Extract the class labels in an array
		String [] trainClasslabel = extractClassLabels(ri);

		//Construct a new data that contains average of random imputed values
		Instances avgdata = avgData(train, trainClasslabel, val);		

		return avgdata;
	}

	//Average EM imputation on test set
	public static Instances avgEMImputationTest (Instances test, int Imputations) throws Exception {
		double [][] val = new double[test.numInstances()][test.numAttributes()-1];
		//find the average of all the imputations
		Instances ri = new Instances(test,test.numInstances());
		for (int k=0;k<Imputations;k++) {
			ri = EMaxImputationTest(test,k);
			//Check
			if (ri.numInstances()!=test.numInstances()) 
				throw new Exception("size mismatch in Testset Avg. EMImputation");
			for (int i=0;i<ri.numInstances();i++) {
				for (int j=0;j<ri.numAttributes()-1;j++) {
					val[i][j] += ri.instance(i).value(j)/Imputations;
				}
			}
		}

		//Extract the class labels in an array
		String [] trainClasslabel = extractClassLabels(ri);

		//Check
		//if (trainClasslabel.length!=ri.numInstances()) 
		//	throw new Exception("size mismatch in Testset EMImputation");

		//Construct a new data that contains average of random imputed values
		Instances avgdata = avgData(test, trainClasslabel, val);		

		return avgdata;
	}

	//Average Bayesian Multiple imputation on train set
	public static Instances avgBMImputation (Instances train, int Imputations) throws Exception {
		double [][] val = new double[train.numInstances()][train.numAttributes()-1];
		//find the average of all the imputations
		Instances ri = new Instances(train, train.numInstances());
		for (int k=0;k<Imputations;k++) {
			ri = BMImputation(train);
			bmifilterArray.add(k, bmifilter);
			for (int i=0;i<ri.numInstances();i++) {
				for (int j=0;j<ri.numAttributes()-1;j++) {
					val[i][j] += ri.instance(i).value(j)/Imputations;
				}
			}
		}

		//Extract the class labels in an array
		String [] trainClasslabel = extractClassLabels(ri);

		//Construct a new data that contains average of random imputed values
		Instances avgdata = avgData(train, trainClasslabel, val);		

		return avgdata;
	}

	//Average Bayesian Multiple imputation on testset
	public static Instances avgBMImputationTest (Instances test, int Imputations) throws Exception {
		double [][] val = new double[test.numInstances()][test.numAttributes()-1];
		//find the average of all the imputations
		Instances ri =  new Instances(test,test.numInstances());
		for (int k=0;k<Imputations;k++) {
			ri = BMImputationTest(test,k);
			for (int i=0;i<ri.numInstances();i++) {
				for (int j=0;j<ri.numAttributes()-1;j++) {
					val[i][j] += ri.instance(i).value(j)/Imputations;
				}
			}
		}

		//Extract the class labels in an array
		String [] trainClasslabel = extractClassLabels(ri);

		//Construct a new data that contains average of random imputed values
		Instances avgdata = avgData(test, trainClasslabel, val);		

		return avgdata;
	}

	//Extract class labels in an array of strings
	public static String [] extractClassLabels (Instances train) {
		String [] trainClasslabel = new String[train.numInstances()];
		for (int i=0;i<train.numInstances();i++) {
			int numVal=(int) train.instance(i).value(train.numAttributes()-1); //Numeric value of class label
			trainClasslabel[i]=train.attribute(train.numAttributes()-1).value(numVal).toString(); //Convert numeric class value to String value
			//System.out.println(trainClasslabel[i]);

		}
		return trainClasslabel;
	}

	//Extract class labels in an array of int
	public static int [] extractClassLabelsInt (Instances train) {
		int [] trainClasslabel = new int[train.numInstances()];
		for (int i=0;i<train.numInstances();i++) {
			trainClasslabel[i]=(int) train.instance(i).value(train.numAttributes()-1);

		}
		return trainClasslabel;
	}

	//Create a new data that contains the average of imputation performed 'n' times
	public static Instances avgData(Instances train, String [] trainClasslabel, double [][] val) {
		Instances avgri = new Instances(train, train.numInstances());
		for (int i=0;i<train.numInstances();i++) {
			//Add the imputed value in a new Instance variable
			Instance temp = new DenseInstance(train.numAttributes());
			temp.setDataset(train);
			//temp.setDataset(train);
			for (int j=0;j<train.numAttributes()-1;j++) {
				temp.setValue(j, val[i][j]);
			}
			temp.setValue(train.numAttributes()-1, trainClasslabel[i]);//Add the class label
			avgri.add(temp);
		}
		return avgri;

	}

	//Random Imputation - find the range of attributes and perform random imputation
	public static Instances randomImputation (Instances data, Instances test) {		
		double[][] minmax = findMinMaxRange(data);
		//Extract min and max values separately of each attribute
		double [] max = new double [data.numAttributes()-1];
		double [] min = new double [data.numAttributes()-1];
		for (int i=0;i<data.numAttributes()-1;i++) {
			max[i]=minmax[i][0];
			min[i]=minmax[i][1];
		}				
		Instances ri = funcRandImputation(test, max, min);

		return ri;
	} //end randImputation

	//Gaussian Random Imputation - find the mean and std deviation of attributes and perform gaussian random imputation
	public static Instances gaussianRandomImputation (Instances data, Instances test, double [] zvalue) {		
		double[][] minmax = findMinMaxRange(data);
		//Extract mean and std deviation values separately of each attribute
		double [] mean = new double [data.numAttributes()-1];
		double [] sdev = new double [data.numAttributes()-1];
		for (int i=0;i<data.numAttributes()-1;i++) {
			mean[i]=minmax[i][2];
			sdev[i]=minmax[i][3];
		}				
		Instances ri = funcGaussianRandImputation(test, mean, sdev, zvalue);

		return ri;
	} //end randImputation

	//Find the maximum and minimum range, mean and std deviation of each attribute
	public static double [][] findMinMaxRange(Instances data) {
		double [][] minmax = new double [data.numAttributes()][4];
		//Find stats of each attribute
		for (int i=0;i<data.numAttributes()-1;i++) {
			Stats s = data.attributeStats(i).numericStats;
			minmax[i][0]=s.max;//max value
			minmax[i][1]=s.min;//min value
			minmax[i][2]=s.mean;//mean value
			minmax[i][3]=s.stdDev;//std deviation
		}
		return minmax;
	}

	//Function to perform random imputation
	public static Instances funcRandImputation(Instances data, double [] max, double [] min) {
		Instances ri = new Instances (data, data.numInstances());
		for (int i=0;i<data.numInstances();i++) {
			Instance a = new DenseInstance(data.numAttributes());
			for(int j=0;j<data.numAttributes()-1;j++) {
				if (data.instance(i).isMissing(j)) {
					double rv = ThreadLocalRandom.current().nextDouble(min[j],max[j]+1);
					//double rv = min[j] + (val *(max[j]-min[j]));
					System.out.print(data.instance(i)+" j="+j+", rv="+rv);
					a.setValue(j, rv);
					System.out.println();
				}
				else 
					a.setValue(j, data.instance(i).value(j));
			}
			a.setValue(data.classAttribute(),data.instance(i).classValue());
			ri.add(i,a);
		}
		return ri;
	} //end funcRandImputation

	//Function to perform Gaussian random imputation
	public static Instances funcGaussianRandImputation(Instances data, double [] mean, double [] sdev, double [] zvalue) {
		Instances ri = new Instances (data, data.numInstances());
		for (int i=0;i<data.numInstances();i++) {
			Instance a = new DenseInstance(data.numAttributes());
			for(int j=0;j<data.numAttributes()-1;j++) {
				if (data.instance(i).isMissing(j)) {					
					//generate a random value between z_min and z_max
					double val = ThreadLocalRandom.current().nextDouble(zvalue[1],zvalue[0]+1);
					double rv = sdev[j]*val + mean[j]; 
					//double rv = min[j] + (val *(max[j]-min[j]));
					//System.out.print(data.instance(i)+" j="+j+", rv="+rv);
					a.setValue(j, rv);
					//System.out.println();
				}
				else 
					a.setValue(j, data.instance(i).value(j));
			}
			a.setValue(data.classAttribute(),data.instance(i).classValue());
			ri.add(i,a);
		}
		return ri;
	} //end funcGaussianRandImputation

	//Create bootstrap samples
	public static Instances bootstrapSamples(Instances newData, int seed) throws Exception {
		Resample re = new Resample();
		re.setNoReplacement(false);
		re.setRandomSeed(seed); //Choose different random everytime
		re.setInputFormat(newData);
		Instances data = Filter.useFilter(newData, re);
		return data;
	}

	public static void printOutput(double [][] val, int [] MissingRatio, String output, String [] methodNames) throws IOException{
		BufferedWriter out = new BufferedWriter(new FileWriter(output));
		out.write(",");
		for(int i=0; i< MissingRatio.length;i++)
			out.write(MissingRatio[i]+",");
		out.newLine();
		DecimalFormat df = new DecimalFormat("#0.000");
		for (int j=0;j<methodNames.length;j++) {
			System.out.print(methodNames[j]+"--");
			out.write(methodNames[j]+",");
			for(int i=0; i< MissingRatio.length;i++) {
				System.out.print(df.format(val[i][j])+" ");
				out.write(df.format(val[i][j])+",");
			}
			System.out.println();
			out.newLine();
		}
		out.close();
	}

} //End of class
