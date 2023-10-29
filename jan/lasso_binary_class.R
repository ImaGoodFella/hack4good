setwd("C:/Users/janwe/Desktop/Uni/ETH/HS2023/Hack4Good/Data")
labels <- read.csv("labels.csv")
data <- read.csv("tf_features_full_narm.csv")

##set label
damage <- as.numeric(labels['extent']>20)


##fix the dataset
data_o <- data[order(data['X']),]
rownames(data_o) <- 1:nrow(data_o)
data_o['damage']<-damage
data_o['damage_type']<-labels['damage']


require(dplyr)

##create train and test sets
set.seed(1)

train <- data_o %>% dplyr::sample_frac(0.70)
test  <- dplyr::anti_join(data_o, train, by = 'X')


test_x <- data.matrix(subset(test, select = -c(X,damage,filename,damage_type) ))
test_y <- data.matrix(test['damage'])
test_yclass <- as.factor(t(test['damage_type']))
train_x <- data.matrix(subset(train, select = -c(X,damage,filename,damage_type) ))
train_y <- data.matrix(train['damage'])
train_yclass <- as.factor(t(train['damage_type']))




require(glmnet)

##lasso regression with cross-validation (took about 10 min)
sFile <- "cv-lasso-climate.rds"
if(!file.exists(sFile)) {
  set.seed(1)
  print(system.time(
    cv.lasso <- cv.glmnet(train_x, train_y, alpha=1, nfolds=10)
  ))
  saveRDS(cv.lasso, sFile)
} else {
  cv.lasso <- readRDS(sFile)
}

##prediction

pred <-  predict(cv.lasso,test_x,lambda="lambda.1se")


##how accurate would prediction be

preds <- as.numeric(pred>0.5)

mean(as.numeric(preds==test_y))


## get loss

require(DPpack)

res <- loss.cross.entropy(pred, test_y)

mean(res,na.rm = TRUE) #0.5103735 


## Classification using random forest

require(randomForest)



sFile <- "rf_climate_class.rds"
if(!file.exists(sFile)) {
  set.seed(1)
  print(system.time(
    fit.rf <- randomForest(train_x,train_yclass)
  ))
  saveRDS(fit.rf, sFile)
} else {
  fit.rf <- readRDS(sFile)
}



predc <- predict(fit.rf,test_x,type="response")

mean(as.numeric(predc==test_yclass)) #0.5368079

## Fit a Random forest using the relevant features extracted using the lasso

require(coefplot)

relevant.features <- extract.coef(cv.lasso)$Coefficient[-1]
test_x_relevant <- data.matrix(subset(test, select = relevant.features))
train_x_relevant <- data.matrix(subset(train, select = relevant.features))

sFile <- "rf_climate_class_relevant.rds"
if(!file.exists(sFile)) {
  set.seed(1)
  print(system.time(
    fit.rf.relevant <- randomForest(train_x_relevant,train_yclass)
  ))
  saveRDS(fit.rf.relevant, sFile)
} else {
  fit.rf.relevant <- readRDS(sFile)
}



predc.rel <- predict(fit.rf.relevant,test_x_relevant,type="response")

mean(as.numeric(predc.rel==test_yclass)) #0.5388184
