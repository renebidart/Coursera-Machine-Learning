Exercise Classification
========================================================




```r
library(doParallel)
```

```
## Loading required package: foreach
## Loading required package: iterators
## Loading required package: parallel
```

```r
cl<-makeCluster(detectCores())
registerDoParallel(cl)

library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
setwd("/Users/rene_b23/desktop/coursera/machine learning")
training1<-as.data.frame(read.csv("./pml-training.csv"))
testing<-as.data.frame(read.csv("./pml-testing.csv"))
```

Create partitions for use with cross validation:

```r
inTrain = createDataPartition(training1$classe, p = 3/4)[[1]]
training = training1[ inTrain,]
testing2 = training1[-inTrain,]
```



```r
##Get info

str(training)
```

```
## 'data.frame':	14718 obs. of  160 variables:
##  $ X                       : int  1 2 3 4 5 7 8 10 11 12 ...
##  $ user_name               : Factor w/ 6 levels "adelmo","carlitos",..: 2 2 2 2 2 2 2 2 2 2 ...
##  $ raw_timestamp_part_1    : int  1323084231 1323084231 1323084231 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 ...
##  $ raw_timestamp_part_2    : int  788290 808298 820366 120339 196328 368296 440390 484434 500302 528316 ...
##  $ cvtd_timestamp          : Factor w/ 20 levels "02/12/2011 13:32",..: 9 9 9 9 9 9 9 9 9 9 ...
##  $ new_window              : Factor w/ 2 levels "no","yes": 1 1 1 1 1 1 1 1 1 1 ...
##  $ num_window              : int  11 11 11 12 12 12 12 12 12 12 ...
##  $ roll_belt               : num  1.41 1.41 1.42 1.48 1.48 1.42 1.42 1.45 1.45 1.43 ...
##  $ pitch_belt              : num  8.07 8.07 8.07 8.05 8.07 8.09 8.13 8.17 8.18 8.18 ...
##  $ yaw_belt                : num  -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 ...
##  $ total_accel_belt        : int  3 3 3 3 3 3 3 3 3 3 ...
##  $ kurtosis_roll_belt      : Factor w/ 397 levels "","-0.016850",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ kurtosis_picth_belt     : Factor w/ 317 levels "","-0.021887",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ kurtosis_yaw_belt       : Factor w/ 2 levels "","#DIV/0!": 1 1 1 1 1 1 1 1 1 1 ...
##  $ skewness_roll_belt      : Factor w/ 395 levels "","-0.003095",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ skewness_roll_belt.1    : Factor w/ 338 levels "","-0.005928",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ skewness_yaw_belt       : Factor w/ 2 levels "","#DIV/0!": 1 1 1 1 1 1 1 1 1 1 ...
##  $ max_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_picth_belt          : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_yaw_belt            : Factor w/ 68 levels "","-0.1","-0.2",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ min_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_pitch_belt          : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_yaw_belt            : Factor w/ 68 levels "","-0.1","-0.2",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ amplitude_roll_belt     : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_pitch_belt    : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_yaw_belt      : Factor w/ 4 levels "","#DIV/0!","0.00",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ var_total_accel_belt    : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_roll_belt        : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_pitch_belt          : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_pitch_belt       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_pitch_belt          : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_yaw_belt            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_yaw_belt         : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_yaw_belt            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ gyros_belt_x            : num  0 0.02 0 0.02 0.02 0.02 0.02 0.03 0.03 0.02 ...
##  $ gyros_belt_y            : num  0 0 0 0 0.02 0 0 0 0 0 ...
##  $ gyros_belt_z            : num  -0.02 -0.02 -0.02 -0.03 -0.02 -0.02 -0.02 0 -0.02 -0.02 ...
##  $ accel_belt_x            : int  -21 -22 -20 -22 -21 -22 -22 -21 -21 -22 ...
##  $ accel_belt_y            : int  4 4 5 3 2 3 4 4 2 2 ...
##  $ accel_belt_z            : int  22 22 23 21 24 21 21 22 23 23 ...
##  $ magnet_belt_x           : int  -3 -7 -2 -6 -6 -4 -2 -3 -5 -2 ...
##  $ magnet_belt_y           : int  599 608 600 604 600 599 603 609 596 602 ...
##  $ magnet_belt_z           : int  -313 -311 -305 -310 -302 -311 -313 -308 -317 -319 ...
##  $ roll_arm                : num  -128 -128 -128 -128 -128 -128 -128 -128 -128 -128 ...
##  $ pitch_arm               : num  22.5 22.5 22.5 22.1 22.1 21.9 21.8 21.6 21.5 21.5 ...
##  $ yaw_arm                 : num  -161 -161 -161 -161 -161 -161 -161 -161 -161 -161 ...
##  $ total_accel_arm         : int  34 34 34 34 34 34 34 34 34 34 ...
##  $ var_accel_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_roll_arm         : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_pitch_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_pitch_arm        : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_pitch_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_yaw_arm             : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_yaw_arm          : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_yaw_arm             : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ gyros_arm_x             : num  0 0.02 0.02 0.02 0 0 0.02 0.02 0.02 0.02 ...
##  $ gyros_arm_y             : num  0 -0.02 -0.02 -0.03 -0.03 -0.03 -0.02 -0.03 -0.03 -0.03 ...
##  $ gyros_arm_z             : num  -0.02 -0.02 -0.02 0.02 0 0 0 -0.02 0 0 ...
##  $ accel_arm_x             : int  -288 -290 -289 -289 -289 -289 -289 -288 -290 -288 ...
##  $ accel_arm_y             : int  109 110 110 111 111 111 111 110 110 111 ...
##  $ accel_arm_z             : int  -123 -125 -126 -123 -123 -125 -124 -124 -123 -123 ...
##  $ magnet_arm_x            : int  -368 -369 -368 -372 -374 -373 -372 -376 -366 -363 ...
##  $ magnet_arm_y            : int  337 337 344 344 337 336 338 334 339 343 ...
##  $ magnet_arm_z            : int  516 513 513 512 506 509 510 516 509 520 ...
##  $ kurtosis_roll_arm       : Factor w/ 330 levels "","-0.02438",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ kurtosis_picth_arm      : Factor w/ 328 levels "","-0.00484",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ kurtosis_yaw_arm        : Factor w/ 395 levels "","-0.01548",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ skewness_roll_arm       : Factor w/ 331 levels "","-0.00051",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ skewness_pitch_arm      : Factor w/ 328 levels "","-0.00184",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ skewness_yaw_arm        : Factor w/ 395 levels "","-0.00311",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ max_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_picth_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_yaw_arm             : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_pitch_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_yaw_arm             : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_roll_arm      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_pitch_arm     : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_yaw_arm       : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ roll_dumbbell           : num  13.1 13.1 12.9 13.4 13.4 ...
##  $ pitch_dumbbell          : num  -70.5 -70.6 -70.3 -70.4 -70.4 ...
##  $ yaw_dumbbell            : num  -84.9 -84.7 -85.1 -84.9 -84.9 ...
##  $ kurtosis_roll_dumbbell  : Factor w/ 398 levels "","-0.0035","-0.0073",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ kurtosis_picth_dumbbell : Factor w/ 401 levels "","-0.0163","-0.0233",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ kurtosis_yaw_dumbbell   : Factor w/ 2 levels "","#DIV/0!": 1 1 1 1 1 1 1 1 1 1 ...
##  $ skewness_roll_dumbbell  : Factor w/ 401 levels "","-0.0082","-0.0096",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ skewness_pitch_dumbbell : Factor w/ 402 levels "","-0.0053","-0.0084",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ skewness_yaw_dumbbell   : Factor w/ 2 levels "","#DIV/0!": 1 1 1 1 1 1 1 1 1 1 ...
##  $ max_roll_dumbbell       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_picth_dumbbell      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_yaw_dumbbell        : Factor w/ 73 levels "","-0.1","-0.2",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ min_roll_dumbbell       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_pitch_dumbbell      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_yaw_dumbbell        : Factor w/ 73 levels "","-0.1","-0.2",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ amplitude_roll_dumbbell : num  NA NA NA NA NA NA NA NA NA NA ...
##   [list output truncated]
```

```r
apply(training, 2, function(x) length(which(is.na(x))))
```

```
##                        X                user_name     raw_timestamp_part_1 
##                        0                        0                        0 
##     raw_timestamp_part_2           cvtd_timestamp               new_window 
##                        0                        0                        0 
##               num_window                roll_belt               pitch_belt 
##                        0                        0                        0 
##                 yaw_belt         total_accel_belt       kurtosis_roll_belt 
##                        0                        0                        0 
##      kurtosis_picth_belt        kurtosis_yaw_belt       skewness_roll_belt 
##                        0                        0                        0 
##     skewness_roll_belt.1        skewness_yaw_belt            max_roll_belt 
##                        0                        0                    14410 
##           max_picth_belt             max_yaw_belt            min_roll_belt 
##                    14410                        0                    14410 
##           min_pitch_belt             min_yaw_belt      amplitude_roll_belt 
##                    14410                        0                    14410 
##     amplitude_pitch_belt       amplitude_yaw_belt     var_total_accel_belt 
##                    14410                        0                    14410 
##            avg_roll_belt         stddev_roll_belt            var_roll_belt 
##                    14410                    14410                    14410 
##           avg_pitch_belt        stddev_pitch_belt           var_pitch_belt 
##                    14410                    14410                    14410 
##             avg_yaw_belt          stddev_yaw_belt             var_yaw_belt 
##                    14410                    14410                    14410 
##             gyros_belt_x             gyros_belt_y             gyros_belt_z 
##                        0                        0                        0 
##             accel_belt_x             accel_belt_y             accel_belt_z 
##                        0                        0                        0 
##            magnet_belt_x            magnet_belt_y            magnet_belt_z 
##                        0                        0                        0 
##                 roll_arm                pitch_arm                  yaw_arm 
##                        0                        0                        0 
##          total_accel_arm            var_accel_arm             avg_roll_arm 
##                        0                    14410                    14410 
##          stddev_roll_arm             var_roll_arm            avg_pitch_arm 
##                    14410                    14410                    14410 
##         stddev_pitch_arm            var_pitch_arm              avg_yaw_arm 
##                    14410                    14410                    14410 
##           stddev_yaw_arm              var_yaw_arm              gyros_arm_x 
##                    14410                    14410                        0 
##              gyros_arm_y              gyros_arm_z              accel_arm_x 
##                        0                        0                        0 
##              accel_arm_y              accel_arm_z             magnet_arm_x 
##                        0                        0                        0 
##             magnet_arm_y             magnet_arm_z        kurtosis_roll_arm 
##                        0                        0                        0 
##       kurtosis_picth_arm         kurtosis_yaw_arm        skewness_roll_arm 
##                        0                        0                        0 
##       skewness_pitch_arm         skewness_yaw_arm             max_roll_arm 
##                        0                        0                    14410 
##            max_picth_arm              max_yaw_arm             min_roll_arm 
##                    14410                    14410                    14410 
##            min_pitch_arm              min_yaw_arm       amplitude_roll_arm 
##                    14410                    14410                    14410 
##      amplitude_pitch_arm        amplitude_yaw_arm            roll_dumbbell 
##                    14410                    14410                        0 
##           pitch_dumbbell             yaw_dumbbell   kurtosis_roll_dumbbell 
##                        0                        0                        0 
##  kurtosis_picth_dumbbell    kurtosis_yaw_dumbbell   skewness_roll_dumbbell 
##                        0                        0                        0 
##  skewness_pitch_dumbbell    skewness_yaw_dumbbell        max_roll_dumbbell 
##                        0                        0                    14410 
##       max_picth_dumbbell         max_yaw_dumbbell        min_roll_dumbbell 
##                    14410                        0                    14410 
##       min_pitch_dumbbell         min_yaw_dumbbell  amplitude_roll_dumbbell 
##                    14410                        0                    14410 
## amplitude_pitch_dumbbell   amplitude_yaw_dumbbell     total_accel_dumbbell 
##                    14410                        0                        0 
##       var_accel_dumbbell        avg_roll_dumbbell     stddev_roll_dumbbell 
##                    14410                    14410                    14410 
##        var_roll_dumbbell       avg_pitch_dumbbell    stddev_pitch_dumbbell 
##                    14410                    14410                    14410 
##       var_pitch_dumbbell         avg_yaw_dumbbell      stddev_yaw_dumbbell 
##                    14410                    14410                    14410 
##         var_yaw_dumbbell         gyros_dumbbell_x         gyros_dumbbell_y 
##                    14410                        0                        0 
##         gyros_dumbbell_z         accel_dumbbell_x         accel_dumbbell_y 
##                        0                        0                        0 
##         accel_dumbbell_z        magnet_dumbbell_x        magnet_dumbbell_y 
##                        0                        0                        0 
##        magnet_dumbbell_z             roll_forearm            pitch_forearm 
##                        0                        0                        0 
##              yaw_forearm    kurtosis_roll_forearm   kurtosis_picth_forearm 
##                        0                        0                        0 
##     kurtosis_yaw_forearm    skewness_roll_forearm   skewness_pitch_forearm 
##                        0                        0                        0 
##     skewness_yaw_forearm         max_roll_forearm        max_picth_forearm 
##                        0                    14410                    14410 
##          max_yaw_forearm         min_roll_forearm        min_pitch_forearm 
##                        0                    14410                    14410 
##          min_yaw_forearm   amplitude_roll_forearm  amplitude_pitch_forearm 
##                        0                    14410                    14410 
##    amplitude_yaw_forearm      total_accel_forearm        var_accel_forearm 
##                        0                        0                    14410 
##         avg_roll_forearm      stddev_roll_forearm         var_roll_forearm 
##                    14410                    14410                    14410 
##        avg_pitch_forearm     stddev_pitch_forearm        var_pitch_forearm 
##                    14410                    14410                    14410 
##          avg_yaw_forearm       stddev_yaw_forearm          var_yaw_forearm 
##                    14410                    14410                    14410 
##          gyros_forearm_x          gyros_forearm_y          gyros_forearm_z 
##                        0                        0                        0 
##          accel_forearm_x          accel_forearm_y          accel_forearm_z 
##                        0                        0                        0 
##         magnet_forearm_x         magnet_forearm_y         magnet_forearm_z 
##                        0                        0                        0 
##                   classe 
##                        0
```

This shows that there are a large number of variables mostly composed of NA's. These columns clearly do not anything to the data so should be removed before any analysis.

The best way to remove these is to remove all columns with near zero variance, as this incluses the NA  columns as well as other unnecessary columns.
Cutoffs were changed to include more data.



```r
x<-nearZeroVar(testing, freqCut = 95/20, uniqueCut = 10, saveMetrics = FALSE)

testSub<-testing[,-x]
trainSub<-training[,-x]

##x<-nearZeroVar(training, freqCut = 95/15, uniqueCut = 10, saveMetrics = FALSE)
```



Create the random forest model and make the prediction on the test set.


```r
model <- train(classe~.,method = "rf", data = trainSub)
```

```
## Loading required package: randomForest
```

```
## Warning: package 'randomForest' was built under R version 3.1.1
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
prediction <- predict(model, newdata = testSub)
```



Use cross validation to see how effective the prediction was:

```r
confusionMatrix(testing2$classe,predict(model,testing2)) 
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1395    0    0    0    0
##          B    0  949    0    0    0
##          C    0    0  855    0    0
##          D    0    0    0  804    0
##          E    0    0    0    0  901
## 
## Overall Statistics
##                                     
##                Accuracy : 1         
##                  95% CI : (0.999, 1)
##     No Information Rate : 0.284     
##     P-Value [Acc > NIR] : <2e-16    
##                                     
##                   Kappa : 1         
##  Mcnemar's Test P-Value : NA        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    1.000    1.000    1.000    1.000
## Specificity             1.000    1.000    1.000    1.000    1.000
## Pos Pred Value          1.000    1.000    1.000    1.000    1.000
## Neg Pred Value          1.000    1.000    1.000    1.000    1.000
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.194    0.174    0.164    0.184
## Detection Prevalence    0.284    0.194    0.174    0.164    0.184
## Balanced Accuracy       1.000    1.000    1.000    1.000    1.000
```
