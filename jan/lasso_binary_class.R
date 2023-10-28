setwd("C:/Users/janwe/Desktop/Uni/ETH/HS2023/Hack4Good/Data")
labels <- read.csv("labels.csv")
data <- read.csv("tf_features_full_narm.csv")

##set label
damage <- as.numeric(labels['extent']>20)


##fix the dataset
data_o <- data[order(data['X']),]
rownames(data_o) <- 1:nrow(data_o)
data_o['damage']<-damage


require(dplyr)

##create train and test sets
set.seed(1)

train <- data_o %>% dplyr::sample_frac(0.70)
test  <- dplyr::anti_join(data_o, train, by = 'X')


test_x <- data.matrix(subset(test, select = -c(X,damage,filename) ))
test_y <- data.matrix(test['damage'])
train_x <- data.matrix(subset(train, select = -c(X,damage,filename) ))
train_y <- data.matrix(train['damage'])




require(glmnet)

##lasso regression with cross-validation (took about 10 min)
sFile <- "cv-lasso-climate.rds"
if(!file.exists(sFile)) {
  set.seed(1)
  print(system.time(
    cv.eln <- cv.glmnet(train_x, train_y, alpha=1, nfolds=10)
  ))
  saveRDS(cv.eln, sFile)
} else {
  cv.eln <- readRDS(sFile)
}

##prediction

pred <-  predict(cv.eln,test_x,lambda="lambda.1se")

require(DPpack)

## get loss

res <- loss.cross.entropy(pred, test_y)

mean(res,na.rm = TRUE)
