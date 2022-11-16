#This is UCI-Bank-Marketing-Dataset
#I have performed 2 tasks on this data set
#1. I have predicted weather customer will subscribe for term Deposit 
#2. Secondly, I have done Customer Segmentation by forming Clusters
#Link for this dataset:https://www.kaggle.com/datasets/alexkataev/bank-marketing-data-set





library(faraway)#for checking Multi co-linearity 
library(caret)
library(e1071)#Required for SVM algo
library(magrittr)#needed for one hot encoding
library(ggcorrplot)#For data Visualization 
library(DMwR)# for SMOTE sampling
library(ff)

# Importing dataset and Renaming Columns with '.' in it -------------------

mydata<-read.csv(file = "bank-additional-full.csv",header = TRUE,sep = ";")

#Renaming columns as it contains '.' in the name
names(mydata)[names(mydata) == 'emp.var.rate'] <- 'emp_var_rate'
names(mydata)[names(mydata) == 'cons.price.idx'] <- 'cons_price_idx'
names(mydata)[names(mydata) == 'cons.conf.idx'] <- 'cons_conf_idx'
names(mydata)[names(mydata) == 'nr.employed'] <- 'nr_employed'
summary(mydata)
x<-cor((mydata[sapply(mydata, is.numeric)])) #Taking correlation of only Numeric data
cor(x)
summary(x)
ggcorrplot(cor(x))   #Plotting correlation Heat map of numeric features  


#Dealing with categorical data

#Giving Ordinal enCoding to Education Feature
word_to_num<-function(word){return(switch(word,"unknown"={0},
                                          "illiterate"={1},
                                          "high.school"={2},
                                          "basic.4y"={3},
                                          "basic.6y"={4} ,
                                          "basic.9y"={5},
                                          "professional.course"={6},
                                          "university.degree"={7}))}


mydata$education=as.numeric(sapply(mydata$education,word_to_num))

#Dummy encoding for categorical features 
dummy<-dummyVars("~ .",data=mydata, fullRank = T)
newdata<-data.frame(predict(dummy,newdata=mydata))
mydataone<-as.data.frame(newdata)
mydataone$yyes=factor(mydataone$yyes,levels = c("1","0"),
                      labels = c("yes","no"))


View(mydataone)


# Data Analysis -----------------------------------------------------------

# Checking VIF Scores for investigating multi-colinearity 
df<-mydata[,c(1,11,12,13,14,16,17,18,19,20,21)]
View(df)
df$y=factor(df$y,levels = c("yes","no"),
            labels = c("1","0"))
df$y=as.numeric(df$y)
lin=lm(df$y~.,data = df)
round(vif(lin),digits = 2) 


#Plotting bar chart of Class imbalance of response variable 
x<-with(mydataone,{print(table(mydataone$yyes))})
x<-as.data.frame(x)
colnames(x) <- c('Class','Freq')
x
ggplot(data=x, aes(x=Class, y=Freq),xlab="Class",ylab="Frequency",title="Class Imbalance") +
  geom_bar(stat="identity", fill="steelblue")+
  geom_text(aes(label=Freq), vjust=1.6, color="white", size=3.5)+
  theme_minimal() 

#Plotting Box plots of different features

boxplot(mydata$age~mydata$marital,main="Boxplots for Age with Marital Status",ylab="Age",ylim=c(0,100),las=1,xlab = "Marital Status")

boxplot(mydata$age~mydata$job,main="Boxplots for Age with Job",xlab="Job",ylab="Age",ylim=c(0,100),las=1, labels=TRUE)

boxplot(mydata$age~mydata$education,main="Boxplots for Age with Education",xlab = "Education",ylab="Age",ylim=c(0,100),las=1)

#Plotting Histogram of features

hist(mydata$emp_var_rate,xlab = "employment variation rate",col = "skyblue",labels = TRUE,ylim = c(0,40000),xlim = c(-4.3,2))


hist(mydata$cons_price_idx,xlab = "Consumer Price Index",col = "skyblue",labels = TRUE, xlim = c(92,95),ylim = c(0,15000))

hist(mydata$cons_conf_idx,xlab = "Consumer Confidence Index",col = "skyblue",labels = TRUE,ylim = c(0,14000),xlim = c(-55,-25) )




# Applying Different Classification Algorithms ----------------------------


#KNN 

#Data Partition into train and Test data using k-fold Cross Validation
train.index <- createDataPartition(mydataone[,"yyes"],p=0.75,list=FALSE)
mydataone.trn <- mydataone[train.index,]
mydataone.tst <- mydataone[-train.index,]

drop <- c("emp_var_rate","euribor3m","nr_employed")
mydataone.trn = mydataone.trn[,!(names(mydataone.trn) %in% drop)]
mydataone.tst = mydataone.tst[,!(names(mydataone.tst) %in% drop)]


ctrl  <- trainControl(method  = "cv",number  = 10) 

fit.cv <- train(yyes ~ ., data = mydataone.trn, method = "knn",
  trControl = ctrl, 
  preProcess = c("center","scale"), 
  tuneGrid =data.frame(k=10))

pred <- predict(fit.cv,mydataone.tst)
confusionMatrix(table(mydataone.tst[,"yyes"],pred))
print(fit.cv)



#Logistic Regression 

fit.cv <- train(yyes ~ ., data = mydataone.trn, method = "glm",
  trControl = ctrl, 

   tuneLength = 10)

pred <- predict(fit.cv,mydataone.tst)
confusionMatrix(table(mydataone.tst[,"yyes"],pred))
print(fit.cv)


#SVM
svm_model<- svm(mydataone.trn$yyes~.,data = mydataone.trn)
summary(svm_model)

#confusion matrix and mis-classification error of model
pred<-predict(svm_model,mydataone.tst)
confusionMatrix(table(mydataone.tst[,"yyes"],pred))




# SMOTE sampling to deal with class imbalance ------------------------------


table(mydataone.trn$yyes)
mydataone.trn$yyes<-as.factor(mydataone.trn$yyes)

finaltarin<-SMOTE(mydataone.trn$yyes~.,mydataone.trn,perc.over = 100,perc.under = 200)
table(finaltarin$yyes)
View(finaltarin)


mydataone.tst$yyes<-as.factor(mydataone.tst$yyes)
fianltest<-SMOTE(mydataone.tst$yyes~.,mydataone.tst,perc.over = 100,perc.under = 200)
table(finaltest$yyes)
View(finaltest)


# Applying KNN after SMOTE Sampling --------------------------------------


ctrl  <- trainControl(method  = "cv",number  = 10) 

fit.cv <- train(yyes ~ ., data =finaltarin , method = "knn",
                trControl = ctrl, 
                preProcess = c("center","scale"), 
                tuneGrid =data.frame(k=10))


pred <- predict(fit.cv,finaltest)
confusionMatrix(table(finaltest[,"yyes"],pred))
print(fit.cv)


# Applying Logistic Regression After SMOTE Sampling ----------------------
ctrl  <- trainControl(method  = "cv",number  = 10) 

fit.cv <- train(yyes ~ ., data =finaltarin , method = "glm",
                trControl = ctrl,
                preProcess = c("center","scale"), 
                tuneLength = 10)


pred <- predict(fit.cv,finaltest)
confusionMatrix(table(finaltest[,"yyes"],pred))
print(fit.cv)


# Applying SVM after SMOTE Sampling --------------------------------------
svm_model<- svm(finaltarin$yyes~.,data = finaltarin)
summary(svm_model)

#confusion matrix and misclassification error of model
pred<-predict(svm_model,finaltest)
confusionMatrix(table(finaltest[,"yyes"],pred))




# Problem 2 : Customer Segmentation ---------------------------------------
#hierarchical clustering ---------
x<-mydata[sapply(mydata, is.numeric)]
View(x)
x<-x[1:1000,]
y<-x

#Normalisation
m<-apply(x,2,mean)
s<-apply(x,2,sd)
x<-scale(x,m,s)

#Calculating Eucledian Distance
distance<-dist(x)
print(distance,digits=3)

#Cluster Dendogram for complete Linkage
hc.c<-hclust(distance)

hc.a<-hclust(distance,method = "average")
plot(hc.c)

#Scree Plot to decide Number of clusters
wss<-(nrow(x)-1)*sum(apply(x[,c(1,2,3,4,10)], 2, var))
for(i in 2:20)wss[i]<-sum(kmeans(x[,c(1,2,3,4,10)],centers = i)$withinss)
plot(1:20,wss,type = "b",xlab = "Number of Clusters",ylab = "Within Group SS")

#cluster Members
member.c<-cutree(hc.c,6)
#member.a<-cutree(hc.a,6)
table(member.c)

#cluster Mean
print(aggregate(x,list(member.c),mean),digits = 2)
aggregate(y,list(member.c),mean)






# K-means -----------------------------------------------------------------
library(factoextra) #For plotting Clusters
#Deciding Number of Clusters

fviz_nbclust(x[,c(1,2,3,4,10)], kmeans, method = "wss")

kc<-kmeans(x[,c(1,2,3,4,10)],4)
kc

fviz_cluster(kc, data = x[,c(1,2,3,4,10)],
             palette = c("#0000FF", "#00FF00" ,"#FF0000","#000000"),#"#2E9FDF","#000000","#E7B800"), 
             geom = "point",
             ellipse.type = "convex", 
             ggtheme = theme_bw()
)

