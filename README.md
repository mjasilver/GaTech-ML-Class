README:

Assignment #2 Machine Learning Georgia Tech

Michael Silver
gtid 902063715
acct msilver9


Code and data can be found on Github here:
	Latest commit:

Files Needed:
	ps2.py

Data Files:
	Titanic:
		gender_submission.csv
		test.csv
		train.csv

To run Neural Networks with Titanic data:
	Place Titanic data in folder "Titanic". Have "Titanic" folder in same directory as code

LOCATION OF DATA ONLINE:
	https://www.kaggle.com/c/titanic/data

To run particular problems (i.e. 4Peaks, SixPeaks, Knapsack):
	All will run by default upon running the appropriate Python file. However, to run a particular algorithm, go to line that says "def main():" and comment out the algorithms that you don't want to run. Leave the target algorithm remaining
	Functions in main() need for neural networks:
		#Neural Net
	    df_submission,df_test,df_train=process_data2() --> preps data for NNs
	    neural_net_RHC(df_submission,df_test,df_train) --> runs Random Hill Climbing. Also runs backprop benchmarking
	    neural_net_SA(df_submission,df_test,df_train)  --> runs Simulated Annealing
    	    neural_net_GA(df_submission,df_test,df_train)  --> runs Genetic Algorithm

		***To see backprop benchmarking, run the Random Hill Climbing algorithm.
