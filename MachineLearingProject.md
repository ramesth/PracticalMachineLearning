Practical Machine Learning : Peer Assessment
========================================================

#### Execuitive summary

This report is the outcome of the analysis performed for  Practical Machine learning course peer assignment.  The predictor is regarding the "manner" the 6 subjects carry out weight lifting exercises. The data used in this project comes from the site here [http://groupware.les.inf.puc-rio.br/har], and are representation of accelerometers readings on the belt, forearm, arm and dumbbell.  The results are represented by five character varibales  ABCDE. It is obviously falls into category of "supervised learning" in terms of machine learning exercise. 

#### Loading, exploring and preprocessing of the data

Let us prepare our environment by loading appropirate libraries. Also set your working directory (not shown here).


```r
library(randomForest);library(caret);library(randomForest) ;library(ElemStatLearn);library(rpart);library(rattle)
```

```
## Warning: package 'randomForest' was built under R version 3.0.3
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```
## Warning: package 'caret' was built under R version 3.0.3
```

```
## Loading required package: lattice
```

```
## Warning: package 'lattice' was built under R version 3.0.3
```

```
## Loading required package: ggplot2
```

```
## Warning: package 'ggplot2' was built under R version 3.0.3
```

```
## Warning: package 'ElemStatLearn' was built under R version 3.0.3
```

```
## Warning: package 'rattle' was built under R version 3.0.3
```

```
## Rattle: A free graphical interface for data mining with R.
## Version 3.4.1 Copyright (c) 2006-2014 Togaware Pty Ltd.
## Type 'rattle()' to shake, rattle, and roll your data.
```

```r
set.seed(9875) ## so that results are reproducible
```
Next, let us download and load the data. It is assumed that files have been downloaded and are resding in the current working directory. 


```r
#file1 <- "pml-training.csv"
#file2 <- "pml-testing.csv"
#download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", destfile=file1)
#download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", destfile=file2)

training <- read.csv("pml-training.csv",row.names=1,na.strings=c("", "NA", "NULL"))
testing <- read.csv("pml-testing.csv",row.names=1,na.strings=c("", "NA", "NULL"))
```
Next, let us explore structure of our data.

```r
## commented out in order to save space
#str(training)
#str(testing)
```
The structure commands (not run and shown here) reveal that there two many NAs and many unneeded variables. 
Next step, removes the data containing NAs. In addition,  remove extraneous columns such as user_name, num_window, time_stamp etc . These  variables will not contribute to our analysis.

```r
training <- training[, colSums(is.na(training)) == 0] 
testing <- testing[, colSums(is.na(testing)) == 0]
training <- training[, c(7:59)] 
testing <- testing[, c(7:59)] 
# examine the number of rows and column each data set contains
dim(training); 
```

```
## [1] 19622    53
```

```r
dim(testing)
```

```
## [1] 20 53
```
#### Model Building

Now we split provided training data into test (30%) and training (70%) data sets. When we are satisfied, we will apply our model to predict supplied "test" data.

```r
inTrain <- createDataPartition(y = training$classe, p=0.70, list=FALSE) 
trainFinal <- training[inTrain,] 
testFinal <- training[-inTrain,]
```
Next, it is time to select a model and train. first let us try rpart method


```r
modelFitRpart <- train(classe ~ ., method = "rpart", data = trainFinal)
```

```
## Loading required namespace: e1071
```

```r
modelPredRpart <- predict(modelFitRpart, newdata=testFinal)
```
Next, let us evaluate how good is the model, and whether we need to modify or change our model.

```r
# verify the model performance

confusionMatrix(modelPredRpart,testFinal$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1535  489  469  438  156
##          B   24  380   29  169  147
##          C  112  270  528  357  291
##          D    0    0    0    0    0
##          E    3    0    0    0  488
## 
## Overall Statistics
##                                           
##                Accuracy : 0.498           
##                  95% CI : (0.4852, 0.5109)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.3436          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9170  0.33363  0.51462   0.0000  0.45102
## Specificity            0.6314  0.92225  0.78802   1.0000  0.99938
## Pos Pred Value         0.4972  0.50734  0.33890      NaN  0.99389
## Neg Pred Value         0.9503  0.85222  0.88491   0.8362  0.88988
## Prevalence             0.2845  0.19354  0.17434   0.1638  0.18386
## Detection Rate         0.2608  0.06457  0.08972   0.0000  0.08292
## Detection Prevalence   0.5246  0.12727  0.26474   0.0000  0.08343
## Balanced Accuracy      0.7742  0.62794  0.65132   0.5000  0.72520
```
As can be seen that this has a very low acccuracy of 49.8. Hence let us use Random Forest with cross validation and preprocessing.

```r
modelFit <- train(classe ~ ., method = "rf", data = trainFinal,preProcess=c("center", "scale"), trControl = trainControl(method = "cv", number = 4))
modelPred <- predict(modelFit, newdata=testFinal)
```
Next, let us evaluate how good is the chosen model, and whether we need to modify or change our model.

```r
# verify the model performance

confusionMatrix(modelPred,testFinal$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1669   12    0    0    0
##          B    4 1124    6    0    1
##          C    0    2 1018    7    2
##          D    0    1    2  956    2
##          E    1    0    0    1 1077
## 
## Overall Statistics
##                                          
##                Accuracy : 0.993          
##                  95% CI : (0.9906, 0.995)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9912         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9970   0.9868   0.9922   0.9917   0.9954
## Specificity            0.9972   0.9977   0.9977   0.9990   0.9996
## Pos Pred Value         0.9929   0.9903   0.9893   0.9948   0.9981
## Neg Pred Value         0.9988   0.9968   0.9984   0.9984   0.9990
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2836   0.1910   0.1730   0.1624   0.1830
## Detection Prevalence   0.2856   0.1929   0.1749   0.1633   0.1833
## Balanced Accuracy      0.9971   0.9923   0.9950   0.9953   0.9975
```
The model has accuracy of 99.3% which is very good for our purpose. 

#### Out of sample Error

For our model "out of sample error"" is 1-0.993=0.007

#### Prediction Results

This section will predict the desired results.


```r
predictions <- predict(modelFit,newdata=testing)
predUsingTesting <- as.character(predictions) 
predUsingTesting
```

```
##  [1] "B" "A" "B" "A" "A" "E" "D" "B" "A" "A" "B" "C" "B" "A" "E" "E" "A"
## [18] "B" "B" "B"
```


```r
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(predUsingTesting)
```
#### Conclusion  

Since the accuracy with Rpart is very low, we discarded the first model.

With accuracy of 99.29% , the Random Forest model prediction outcomes are B A B A A E D B A A B C B A E E A B B B.
