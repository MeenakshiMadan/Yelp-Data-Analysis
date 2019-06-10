####IMporting the packages
library(mice)
library(VIM)
library(lattice)
library(ggplot2)
library(dplyr)
library(caret)
library("randomForest")
library(lubridate)
library(LiblineaR)
library(pROC)

####Reading the file ######
Data_missing <- read.csv("Untitled Folder/combined_yelp_new.csv",header=TRUE, sep=",")
View(Data_missing)

###Removing the records where restaurants is not open
Data_missing <- Data_missing[Data_missing$is_open == "1" ,]

colnames(Data_missing)

####Finding Missing Data in Columns
percentage_null<-NULL
for(i in 1:51){
  percentage_null[i]<-(sum(is.na(Data_missing[,i]))/58567)*100
}

Column_Names<-colnames(Data_missing)
#xyz<-xyz[-c(which(abc<="50.5"))]
#abc<-abc[-c(which(abc<="50.5"))]

combined_names<-as.data.frame(cbind(Column_Names,percentage_null))
combined_names
####THere are few columns which have 0 missing values and some have more than 99% missing values
###here i will be dropping all the columns which have more than 50% missing data , as the 
##mice package will not be able to do justtice imputing the data

Data_yelp<-Data_missing[-c(which(percentage_null>="50"))]
colnames(Data_yelp)
data_yelp<-as.data.frame(Data_yelp)
#####Also dropping the columns which have categorical data 
###or columns which will not help in prediction are removed
data_yelp<-select (data_yelp,-c(RestaurantsCounterService,Open24Hours,AgesAllowed,Smoking,BYOB,Corkage,BestNights
                                ,RestaurantsAttire,Alcohol,WiFi,NoiseLevel,Categories,Name
                                ,Business_id,Categories,Latitude,longitude,is_open,state,city,postal_code,BYOBCorkage
                                ,Open24Hours))

colnames(data_yelp)
#####Dividing the dataset into parts based on the type of data in columns
data_part<-data_yelp[1:2]
data_part1<-data_yelp[3:10]
data_part2<-data_yelp[11:18]
data_part3<-data_yelp[19:23]


###### imputing missing data
###converting the columns with no missing values to data frame
data_part<-as.data.frame(data_part)


#####imputation of 1st part and then picking the final result
tempData1 <- mice(data_part1,m=3,maxit=5,meth='pmm',seed=100)
summary(tempData1)
complete_data1<-complete(tempData1,1)
summary(complete_data1)
sum(is.na(complete_data1[,1:8]))
complete_data1<-as.data.frame(complete_data1)


#####imputation of 2nd part and then picking the final result
tempData2 <- mice(data_part2,m=3,maxit=5,meth='pmm',seed=100)
summary(tempData2)
complete_data2<-complete(tempData2,1)
summary(complete_data2)
sum(is.na(complete_data2[,1:8]))
complete_data2<-as.data.frame(complete_data2)

#####imputation of 3rd part and then picking the final result
tempData3 <- mice(data_part3,m=3,maxit=5,meth='pmm',seed=100)
summary(tempData3)
complete_data3<-complete(tempData3,1)
summary(complete_data3)
sum(is.na(complete_data3[,1:4]))
complete_data3<-as.data.frame(complete_data3)

####combining all the parts from imputations and converting to data frame
data_no_missing_values<-cbind(data_part,complete_data1,complete_data2,complete_data3)
data_no_missing_values<-as.data.frame(data_no_missing_values)


####checking the null values
sum(is.na(data_no_missing_values[,1:23]))


#####Applying the transformations
new_df<-as.data.frame(data_no_missing_values)

# making the star rating into binary classifcation as a good or bad resturant
new_df$Resturant_Type[new_df$stars>=3]<-"Good"
new_df$Resturant_Type[new_df$stars<3]<-"Bad"

# Character columns to factors to be used for random forest
new_df[sapply(new_df, is.character)] <- lapply(new_df[sapply(new_df, is.character)],  as.factor)


new_df$GoodForKids<-as.factor(new_df$GoodForKids)
new_df$GoodForDessert<-as.factor(new_df$GoodForDessert)
new_df$GoodForLatenight<-as.factor(new_df$GoodForLatenight)
new_df$GoodForDinner<-as.factor(new_df$GoodForDinner)
new_df$GoodForBrunch<-as.factor(new_df$GoodForBrunch)
new_df$GoodForBreakfast<-as.factor(new_df$GoodForBreakfast)
new_df$GarageParking<-as.factor(new_df$GarageParking)
new_df$StreetParking<-as.factor(new_df$StreetParking)
new_df$Validated<-as.factor(new_df$Validated)
new_df$LotParking<-as.factor(new_df$LotParking)
new_df$ValetParking<-as.factor(new_df$ValetParking)
new_df$Caters<-as.factor(new_df$Caters)
new_df$RestaurantsTableService<-as.factor(new_df$RestaurantsTableService)
new_df$OutdoorSeating<-as.factor(new_df$OutdoorSeating)
new_df$BikeParking<-as.factor(new_df$BikeParking)
new_df$HasTV<-as.factor(new_df$HasTV)
new_df$RestaurantsGoodForGroups<-as.factor(new_df$RestaurantsGoodForGroups)
new_df$RestaurantsDelivery<-as.factor(new_df$RestaurantsDelivery)
new_df$BusinessAcceptsCreditCards<-as.factor(new_df$BusinessAcceptsCreditCards)

new_df$Resturant_Type<-as.factor(new_df$Resturant_Type)

####removing the star value
new_df<-select (new_df,-c(stars))
##new_df<-select (new_df,-c(Validated))
##new_df<-select (new_df,-c(BusinessAcceptsCreditCards))
##new_df<-select (new_df,-c(OutdoorSeating))
# spliting the data into training and test set 
# Training Set : Validation Set = 70 : 30 (random)
set.seed(100)
train <- sample(nrow(new_df), 0.8*nrow(new_df), replace = FALSE)
TrainingSet <- new_df[train,]
ValidationSet <- new_df[-train,]
summary(TrainingSet)
summary(ValidationSet)

###########Random Forest Model 
model_rf<- randomForest(Resturant_Type~ . , data=TrainingSet, importance=TRUE, ntree=100,prOximity=TRUE,
                  na.action=na.roughfix)
####printing the details for random forest for the first run
model_rf
varImpPlot(model_rf,main='Variable ImportancePlot :Model1',pch=16)

#Printing the variable importance plot
imp_1<-importance(model_rf)

library('e1071')
predict_rf_model1 <- predict(model_rf, ValidationSet, type = "class")
confusion_matrix1 <- confusionMatrix(predict_rf_model1,ValidationSet$Resturant_Type)
confusion_matrix1
varImp(model_rf)

####removing the review count and training the data 
new_df1<-select (new_df1,-c(review_count))


# spliting the data into training and test set 
# Training Set : Validation Set = 70 : 30 (random)
set.seed(100)
train1 <- sample(nrow(new_df1), 0.8*nrow(new_df1), replace = FALSE)
TrainingSet1 <- new_df[train1,]
ValidationSet1 <- new_df[-train1,]
summary(TrainingSet1)
summary(ValidationSet1)

###########Random Forest Model 
model_rf2<- randomForest(Resturant_Type~ . , data=TrainingSet1, importance=TRUE, ntree=300,prOximity=TRUE,
                        na.action=na.roughfix)
####printing the details for random forest for the first run
model_rf2
varImpPlot(model_rf2,main='Variable ImportancePlot :Model2',pch=16)

#Printing the variable importance plot
imp_2<-importance(model_rf2)

library('e1071')
predict_rf_model2 <- predict(model_rf2, ValidationSet1, type = "class")
confusion_matrix2 <- confusionMatrix(predict_rf_model2,ValidationSet1$Resturant_Type)
confusion_matrix2
varImp(model_rf2)


####Applying KNN##################################
knn_df<-as.data.frame(new_df)

colnames(knn_df)

knn_df$GoodForKids<-as.factor(knn_df$GoodForKids)
knn_df$GoodForDessert<-as.factor(knn_df$GoodForDessert)
knn_df$GoodForLatenight<-as.factor(knn_df$GoodForLatenight)
knn_df$GoodForDinner<-as.factor(knn_df$GoodForDinner)
knn_df$GoodForBrunch<-as.factor(knn_df$GoodForBrunch)
knn_df$GoodForBreakfast<-as.factor(knn_df$GoodForBreakfast)
knn_df$GarageParking<-as.factor(knn_df$GarageParking)
knn_df$StreetParking<-as.factor(knn_df$StreetParking)
knn_df$LotParking<-as.factor(knn_df$LotParking)
knn_df$ValetParking<-as.factor(knn_df$ValetParking)
knn_df$Caters<-as.factor(knn_df$Caters)
knn_df$RestaurantsTableService<-as.factor(knn_df$RestaurantsTableService)
knn_df$OutdoorSeating<-as.factor(knn_df$OutdoorSeating)
knn_df$BikeParking<-as.factor(knn_df$BikeParking)
knn_df$HasTV<-as.factor(knn_df$HasTV)
knn_df$RestaurantsGoodForGroups<-as.factor(knn_df$RestaurantsGoodForGroups)
knn_df$RestaurantsDelivery<-as.factor(knn_df$RestaurantsDelivery)
knn_df$BusinessAcceptsCreditCards<-as.factor(knn_df$BusinessAcceptsCreditCards)
knn_df$Resturant_Type<-as.factor(knn_df$Resturant_Type)
####Removing the unwanted columns
knn_df<-select (knn_df,-c(Validated))

intrain <- createDataPartition(y = knn_df$Resturant_Type, p =0.80, list = FALSE)
trainx <- knn_df[intrain,]
testx <- knn_df[-intrain,]


training.x<- trainx[,-22]
training.y <- trainx[,22]
testing.x<- testx[,-22]
testing.y  <- testx[,22]



training.y = as.factor(training.y)
testing.y = as.factor(testing.y)

########## KNN model ###########################

set.seed(130)

ctrl<- trainControl(method = "repeatedcv", number = 5, repeats = 3)

knnFit <- train(x = training.x , y = training.y,
                method = "knn",
                preProc = c("center", "scale"),
                tuneLength = 20,
                trControl = ctrl)

knnFit

####### KNN performance #############

knn.prediction<- predict(knnFit, testing.x)

knn.results <- data.frame(obs = testing.y, pred = knn.prediction)

knn.summary<-defaultSummary(knn.results)
knn.summary

confusionMatrix(knn.prediction,testing.y, positive = "Good")

knn.varImp <-varImp(knnFit)

plot(knn.varImp)
