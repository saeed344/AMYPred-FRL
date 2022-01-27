  ##AMYPred-FRL

AMYPred-FRL is a novel approach for accurate prediction of amyloid proteins by using feature representation learning
In this research study we combined six well-known ML algorithms (extremely randomized tree, extreme gradient boosting, k-nearest neighbor, logistic regression, random forest, and support vector machine) with ten different sequence-based feature descriptors to generate 60 probabilistic features (PFs), as opposed to state-of-the-art methods developed by the single feature-based approach. The logistic regression recursive feature elimination (LR-RFE) method was used to find the optimal m number of 60 PFs in order to improve the predictive performance. Finally, using the meta-predictor approach, the 20 selected PFs were fed into a logistic regression method to create the final hybrid model (AMYPred-FRL).
###AMYPred-FRL uses the following dependencies:
Installation

Download AMYPred-FRL by

git clone https://github.com/saeed344/AMYPred-FRL

Installation has been tested in OS win 10 with Python 3.8.3

Since the package is written in python 3.8.3, python 3.8.3 with the pip tool must be installed first.
 AMYPred-FRL uses the following dependencies: numpy, scipy, scikit-learn, pandas, Xgboost 

You can install these packages first, by the following commands:

pip install numpy

pip install scipy

pip install scikit-learn

pip install pandas

pip install Xgboost

###Guiding principles: 

**The dataset file contains  TR_P_132.fasta, TR_N_305.fasta, TS_P_33.fasta, TS_N_77.fasta.

script test
#####################
To check whether the project can work normally, we can run

Stand_alone_AmyPredFRL using pycharm

To check blind dataset sequnces try Blind_test.py

 



