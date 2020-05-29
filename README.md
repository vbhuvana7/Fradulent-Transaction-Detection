# Fradulent-Transaction-Detection

Introduction
In the global financial industry, the risk of payment fraud is the number one concern for 44% of the financial services firms. Additionally, it is estimated that the average loss due to mobile and online transaction frauds amount to $ 22 Billion in 2019 and is expected to rise to an estimated $48 Billion by the year 2023. With increased digitalization of payment related services around the world, there is growing need to address the nemesis -- the payment fraud. 

The goal of this project is to be able to identify fraud transactions by analyzing customer transactions, identifying fraud patterns and be able to classify transactions are fraud or not fraud with utmost accuracy. The project also aims to evaluate the average loss that a firm may incur due to misclassification and minimize the overall loss through threshold sensitivity analysis that yield better classification confidence. In essence, it aims to optimize the Cost function associated with the model and identify the best classification proportion.


# Threshold sensitivity analysis
Threshold is a value where all values equal or greater than the threshold are mapped to one class and all other values are mapped to another class.Most of the algorithms use 0.5 (probability) as threshold.The goal is to find out optimal threshold by performinng sensitivity analysis

# Cost Matrix:
It is defined as the cost incurred by the company for misclassifying transactions.
There are two types of misclassification (Type I and type II errors). Type I error is predicting that there is no
event when the event is actually present. Type II error is predicting that there is an event when
the event is not actually present. Both the above errors will have some cost associated. The
objective is to minimize these errors by selecting an optimal threshold.
We calculated the average transaction amount of all fraudulent transactions from the dataset is
70$.Hence the Average loss due to misclassification amounted to ~ $ 70.
This is used as the weightage for false-negatives.
