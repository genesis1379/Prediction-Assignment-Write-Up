# Practical Machine Learning Project
Jeremy  
#1.Introduction
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).


The goal of your project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. You may use any of the other variables to predict with. You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases.

<br>

#2.Data Processing

###2.1 Load the relevant libraries and set seed for reproducibility


```r
set.seed(1)
require(caret)
require(randomForest)
```

###2.2 Reading the data

```r
training <- read.csv("pml-training.csv", na.strings = c("NA",""))
testing <- read.csv("pml-testing.csv", na.strings = c("NA",""))
```
The dataset has approximately 20,000 observations and 160 variables.

###2.3 Data Cleaning
####2.3.1 Removing NA variables
Many of the variables have more than 95% NA values so we will get rid of them, this will allow us to work with a complete set of data that is more reliable.


```r
train_data <- training[,colSums(is.na(training)) == 0]
test_data<- testing[,colSums(is.na(testing)) == 0]
```
####2.3.2 Removing unrelated variables
On second glance, the data suggests that first 7 variables are unrelated to the dataset hence we will remove them, this left us with 53 variables.


```r
head(train_data[,1:10])
```

```
##   X user_name raw_timestamp_part_1 raw_timestamp_part_2   cvtd_timestamp
## 1 1  carlitos           1323084231               788290 05/12/2011 11:23
## 2 2  carlitos           1323084231               808298 05/12/2011 11:23
## 3 3  carlitos           1323084231               820366 05/12/2011 11:23
## 4 4  carlitos           1323084232               120339 05/12/2011 11:23
## 5 5  carlitos           1323084232               196328 05/12/2011 11:23
## 6 6  carlitos           1323084232               304277 05/12/2011 11:23
##   new_window num_window roll_belt pitch_belt yaw_belt
## 1         no         11      1.41       8.07    -94.4
## 2         no         11      1.41       8.07    -94.4
## 3         no         11      1.42       8.07    -94.4
## 4         no         12      1.48       8.05    -94.4
## 5         no         12      1.48       8.07    -94.4
## 6         no         12      1.45       8.06    -94.4
```


```r
train_data <- train_data[, -c(1:7)]
test_data <- test_data[, -c(1:7)]
```

###2.4 Data Partition

We need to partition the training data further into training and validation set, usually in 70%:30% split, this allows us to train and optimise our model.


```r
inTrain <- createDataPartition(train_data$classe, p = 0.7, list = FALSE)
train <- train_data[inTrain, ]
valid <- train_data[-inTrain, ]
```

<br>


#3 Prediction Model

###3.1 Modelling approach

We have adopted the Random Forest approach as the author of the course advocates that it is the most used and accurate algorithm. We have also chosen the k-fold cross validation, and instead of using the default k=10 in the trainControl function, we set it to k=5 to cut down the computing processing time.


```r
Control <- trainControl(method = "cv", number = 5)
fit <- train(classe ~.,data = train, method="rf", trControl = Control)
print(fit$finalModel)
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 2
## 
##         OOB estimate of  error rate: 0.68%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3903    2    0    0    1 0.0007680492
## B   17 2635    6    0    0 0.0086531226
## C    0   18 2375    3    0 0.0087646077
## D    0    0   39 2212    1 0.0177619893
## E    0    0    2    4 2519 0.0023762376
```

The output above shows that out-of-bag error rate is 0.7%, this implies that our out-of-sample error would be similar. Note the output suggests a very promising model with high accuracy.

<br>

###3.2 Cross validation and Out-of-Sample Error Rate

This section calculates the actual out-of-sample error rate on the validation data set, using the random forest model built previously.


```r
result <- predict(fit, valid)
confusionMatrix(result, valid$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1673    2    0    0    0
##          B    1 1137   12    0    0
##          C    0    0 1012   18    0
##          D    0    0    2  946    1
##          E    0    0    0    0 1081
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9939          
##                  95% CI : (0.9915, 0.9957)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9923          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9994   0.9982   0.9864   0.9813   0.9991
## Specificity            0.9995   0.9973   0.9963   0.9994   1.0000
## Pos Pred Value         0.9988   0.9887   0.9825   0.9968   1.0000
## Neg Pred Value         0.9998   0.9996   0.9971   0.9964   0.9998
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2843   0.1932   0.1720   0.1607   0.1837
## Detection Prevalence   0.2846   0.1954   0.1750   0.1613   0.1837
## Balanced Accuracy      0.9995   0.9978   0.9913   0.9904   0.9995
```

This output above shows a very good model with out-of-sample accuracy of 99.2%, implying an estimated out-of-sample error rate of 0.8%. We are comfortable with the the random forest model and will not explore any other approaches.

<br>

#4 Prediction on testing dataset

We now use the random forests model to predict the outcome variable classe for the testing dataset.

```r
(predict(fit, test_data))
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

This model predicted the 20 test cases with 100% accuracy from the Course Project Prediction Quiz, hence this concludes the assignment.

<br>
