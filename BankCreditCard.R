



library(data.table)
library(reshape2)
library(randomForest)
library(party)    # FOr decision tree 
library(rpart)    # for Rpart 
library(rpart.plot) #for Rpart plot
library(lattice)  # Used for Data Visualization
require(caret)    # for data pre-processing
library(pROC) 
library(corrplot)  # for correlation plot
library(e1071)    # for ROC curve
library(RColorBrewer)
 

BankCreditCard = read.csv("E:/DSP 17/pratice paper/Bank Credit card - Default prediction Dataset/BankCreditCard.csv")
View(BankCreditCard)
summary(BankCreditCard)

BankCreditCard1 = BankCreditCard[,-(1)]   #remove colomn in onther way.
View(BankCreditCard1)
summary(BankCreditCard1)
             

              #convert factor form


BankCreditCard1$Gender = as.factor(BankCreditCard1$Gender)
BankCreditCard1$Academic_Qualification = as.factor(BankCreditCard1$Academic_Qualification)
BankCreditCard1$Marital = as.factor(BankCreditCard1$Marital)
BankCreditCard1$Default_Payment = as.factor(BankCreditCard1$Default_Payment)
       

                     #boxplot diagram 

boxplot(BankCreditCard1$Credit_Amount ~ BankCreditCard1$Default_Payment, main ="credit_Amount")

boxplot(BankCreditCard1$Jan_Bill_Amount ~ BankCreditCard1$Default_Payment, main ="Jan_Bill_Amount")

boxplot(BankCreditCard1$Feb_Bill_Amount ~ BankCreditCard1$Default_Payment, main ="Feb_Bill_Amount")

boxplot(BankCreditCard1$March_Bill_Amount ~ BankCreditCard1$Default_Payment, main = "March_Bill_Amount")

boxplot(BankCreditCard1$April_Bill_Amount ~ BankCreditCard1$Default_Payment, main ="April_Bill_Amount")

boxplot(BankCreditCard1$May_Bill_Amount ~ BankCreditCard1$Default_Payment, main = "May_Bill_Amount")

boxplot(BankCreditCard1$June_Bill_Amount ~ BankCreditCard1$Default_Payment, main = "June_Bill_Amount")

boxplot(BankCreditCard1$Previous_Payment_Jan ~ BankCreditCard1$Default_Payment, main = "Previous_Payment_Jan")

boxplot(BankCreditCard1$Previous_Payment_Feb ~ BankCreditCard1$Default_Payment, main ="Previous_Payment_Feb")

boxplot(BankCreditCard1$Previous_Payment_March ~ BankCreditCard1$Default_Payment, main ="Previous_Payment_March")

boxplot(BankCreditCard1$Previous_Payment_April ~ BankCreditCard1$Default_Payment, main ="Previous_Payment_April")

boxplot(BankCreditCard1$Previous_Payment_May ~ BankCreditCard1$Default_Payment, main ="Previous_Payment_May" )

boxplot(BankCreditCard1$Previous_Payment_June ~ BankCreditCard1$Default_Payment, main = "Previous_Payment_June" )

         #Checking correlation

cor(BankCreditCard1[,5:23])
corrplot(cor(BankCreditCard1[,5:23]), method="circle")

cor(BankCreditCard[,1:24])
corrplot(cor(BankCreditCard[1:24]), method = "circle")

          # Creating train and test samples
set.seed(1234)
split = createDataPartition(BankCreditCard1$Default_Payment, p = .80,list = FALSE, times = 1)
trainSplit = BankCreditCard1[ split,]
testSplit = BankCreditCard1[-split,]

trainSplit$Default_Payment = as.factor( trainSplit$Default_Payment)
testSplit$Default_Payment = as.factor(testSplit$Default_Payment)

#Check for the event rate
prop.table(table(trainSplit$Default_Payment))
prop.table(table(testSplit$Default_Payment))


# DecisionTree using rpart algorithm

DTree = rpart(Default_Payment ~ ., data = trainSplit)
rpart.plot(DTree)

print(DTree)
summary(DTree)

prp(DTree)
plotcp(DTree)
printcp(DTree)

           #Checking COnfusion matrix on Train data

predtrain = predict(DTree,trainSplit,type = "class" )
confusionMatrix(predtrain,trainSplit$Default_Payment)

        #Checking COnfusion matrix on Test data
predtest = predict(DTree,testSplit, type = "class" )
confusionMatrix(predtest,testSplit$Default_Payment)

# roc(testdata, prediction)
auctrainSplit = roc(as.numeric(trainSplit$Default_Payment), as.numeric(predtrain))
auctestSplit= roc(as.numeric(testSplit$Default_Payment), as.numeric(predtest))
print(auctrainSplit)
print(auctestSplit)


plot(auctrainSplit, ylim=c(0,1), print.thres=TRUE, main=paste('AUC:',round(auctrainSplit$auc[[1]],3)),col = 'blue')



plot(auctestSplit, ylim=c(0,1), print.thres=TRUE, main=paste('AUC:',round(auctestSplit$auc[[1]],3)),col = 'blue')
 

#logistic regression 
logit_BankCreditCard1 = glm(Default_Payment ~.,family = binomial(logit), data = trainSplit)


Default_Paymentpredictions = predict(logit_BankCreditCard1, newdata = testSplit, type = 'response')


logit_confusion =  table(testSplit$Default_Payment, Default_Paymentpredictions>0.5)
logit_confusion

      #accracy
accuracy = sum(diag(logit_confusion))/sum(logit_confusion)
accuracy

Default_Paymentpredictions = data.frame("Probability" = predict(logit_BankCreditCard1, testSplit))

logit_RCRTest = prediction(Default_Paymentpredictions$Probability, testSplit$Default_Payment)

logit_ROCRTestperformance = performance(logit_RCRTest, "tpr", "fpr")

plot(logit_ROCRTestperformance,main="Logistic Regression ROC Curve")

logit_auc <- paste(c("Logistic Regression AUC ="),round(as.numeric(performance(logit_RCRTest,"auc")@y.values),digits=2),sep="")

legend("topleft",logit_auc, bty="n")

