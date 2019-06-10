####Importing all the libraries
library(readxl)
library("ggplot2")
library("tidyr")
library("caret")
library("randomForest")


####Reading the file in R
####trying to see the model performance without any imputations
combined_yelp <- read.csv("Untitled Folder/sample_data.csv",header=TRUE, sep=",")
View(combined_yelp)
typeof(combined_yelp)
####Converting this intoa dataframe
df <- as.data.frame(combined_yelp)

library(dplyr)
#####Applying the transformations
df <- head(combined_yelp, n=1092)
df=as.data.frame(df, stringsAsFactors=T)

# Character columns to factors to be used for random forest
df[sapply(df, is.character)] <- lapply(df[sapply(df, is.character)],  as.factor)

# making factors to be numeric
df[sapply(df, is.factor)] <- lapply(df[sapply(df, is.factor)],  as.numeric)
# Since star rating is what we want to predict , making this as a factor
df$stars <- as.factor(df$stars)




# Diving the dataset into training and test
training <- createDataPartition(df$stars, p=0.65, list=F)
trainData <- df[training, ]
crossValidationData <- df[-training, ]
dim(trainData)

# fitting the random forest model 
rf<- randomForest(stars~ . , data=trainData, importance=TRUE, ntree=300,prOximity=TRUE,
                  na.action=na.roughfix)
####printing the details for random forest for the first run
rf
varImpPlot(rf,main='Variable ImportancePlot :Model1',pch=16)

#Printing the variable importance plot
imp<-importance(rf)
vars<-dimnames(imp)[[1]]
imp<-data.frame(vars=vars,imp=as.numeric(imp[,1]))
imp<-imp[order(imp$imp,decreasing=T),]


selected_data<-c(as.character(imp[1:36,1]),'stars')
selected_data
####retraining the random forest model
rf_2<-randomForest(stars~.,data=trainData[,selected_data],replace=T,ntree=500,prOximity=TRUE,
                   na.action=na.roughfix)
varImpPlot(rf_2,main='Variable Importance : Final Model',pch=16,col='blue')

library('e1071')
predict_rf <- predict(rf_2, crossValidationData, type = "class")
confusion_matrix <- confusionMatrix(predict_rf,crossValidationData$stars)
 



