setwd(utils::getSrcDirectory(function(){})[1])
setwd("..")

if(!require(dplyr)){
  install.packages('dplyr')
  require(dplyr)
}
if(!require(glmnet)){
  install.packages('glmnet')
  require(glmnet)
}
if(!require(coefplot)){
  install.packages('coefplot')
  require(coefplot)
}

##Create training dataset

labels <- read.csv("data/labels.csv")
data <- read.csv("data/tf_features_full_narm.csv")

labels['is_damage'] <- as.numeric(labels['extent']>20)

data_o <- dplyr::inner_join(data, labels, by = 'filename')

X <- data.matrix(data_o %>% select(-c(X,colnames(labels))))
y <- data.matrix(data_o['is_damage'])


## Do Lasso Regression

sFile <- "cv-lasso-climate.rds"
if(!file.exists(sFile)) {
  set.seed(1)
  print(system.time(
    cv.lasso <- cv.glmnet(X, y, alpha=1, nfolds=10)
  ))
  saveRDS(cv.lasso, sFile)
} else {
  cv.lasso <- readRDS(sFile)
}


## Get all features with non zero coefficients

relevant.features <- extract.coef(cv.lasso)$Coefficient[-1]
write.csv(relevant.features, "data/relevant_featurestest.csv")
