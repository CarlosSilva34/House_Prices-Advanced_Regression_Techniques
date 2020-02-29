library("tidyverse")
library("caret")
library("caretEnsemble")
library("kernlab")
library("Matrix")
library("skimr")

library("xgboost")

setwd("C:/Users/Utilizador/Desktop/kaggle/House_Prices-Advanced_Regression_Techniques")
set.seed(4561)

# Load data
db1 <- read.csv('train.csv')
db2 <- read.csv('test.csv')
db2$SalePrice <- as.integer(NA)
hp <- rbind(db1, db2)
remove(db1); remove(db2)

# Overview of the Data
skim(hp)

# Fix variable name starts with number

names(hp)[44:45] <- c("FirstFLSF","SecFLSF")
names(hp)[70] <- c("ThreeSnPorch")

# Fix typos in factorial levels

library(forcats)
count(hp, RoofMatl)
hp <- hp %>%
        mutate(RoofMatl = fct_collapse(RoofMatl, "TarGrv" = "Tar&Grv"))


count(hp, Exterior1st)
hp <- hp %>%
        mutate(Exterior1st = fct_collapse(Exterior1st, "WdSdng" = "Wd Sdng"))

count(hp, Exterior2nd)
hp <- hp %>%
        mutate(Exterior2nd = fct_collapse(Exterior2nd, "BrkComm" = "Brk Cmn", "WdSdng" = "Wd Sdng", "WdShng" = "Wd Shng"))


# Fix typo of GarageYrBlt

summary(hp$GarageYrBlt)
hp <- hp %>% 
        mutate(GarageYrBlt = ifelse(GarageYrBlt > 2010, 2007, GarageYrBlt))


# remove Outliers

ggplot(hp, aes(x=GrLivArea, y=SalePrice)) +
        geom_point() +
        geom_smooth(method=lm, se=FALSE) +
        geom_text(data=hp[hp$GrLivArea>4500,], mapping=aes(label=Id), vjust=1.5, col = "blue") +
        xlab("above grade living area") +
        ylab("sale price")

outlier <- c(524, 1299, 463, 633, 1325, 31, 971)
hp<- hp[!hp$Id %in% outlier, ]

# Feature engineering

hp <- hp  %>%
        mutate(withBsmt = as.numeric(TotalBsmtSF!=0),
               with2ndFloor= as.numeric(SecFLSF>0),
               withPool = as.numeric(PoolArea!=0),
               withPorch = as.numeric((OpenPorchSF+EnclosedPorch+ThreeSnPorch+ScreenPorch)!=0),
               hasRemod = as.numeric(YearRemodAdd != YearBuilt),
               withFireplace = as.numeric(Fireplaces >0),
               isNew = as.numeric(YrSold == YearBuilt),
               totalSF = GrLivArea + TotalBsmtSF,
               totalBath = FullBath + 0.5*HalfBath + 0.5*BsmtHalfBath + BsmtFullBath,
               age = YrSold - YearRemodAdd,
               yearsRemodeled = ifelse(YearRemodAdd == YearBuilt, YrSold < YearRemodAdd, 0),
               YrSold = factor(YrSold),
               MoSold = factor(MoSold),
               MSSubClass = factor(MSSubClass),
               OverallQual = factor(OverallQual),
               OverallCond = factor(OverallCond),
               totalPorchSF = OpenPorchSF + EnclosedPorch + ThreeSnPorch + ScreenPorch,
               withbsmtBath = as.numeric(BsmtHalfBath + BsmtFullBath!=0),
               isBsmtUnf = as.numeric(TotalBsmtSF == BsmtUnfSF)
        )

# Data preprocessing - Use recipe to do preprocessing

## split the data
hp_train <- hp[!is.na(hp$SalePrice),]
hp_test <- hp[is.na(hp$SalePrice),]
set.seed(2123)

## define the recipe
model_recipe <- recipe(SalePrice ~ ., data = hp_train) %>% 
        update_role(Id, new_role = "id var") %>%
        step_knnimpute(all_predictors()) %>%
        step_dummy(all_predictors(), -all_numeric()) %>%
        step_BoxCox(all_predictors()) %>%
        step_center(all_predictors())  %>%
        step_scale(all_predictors()) %>%
        step_zv(all_predictors()) %>%
        step_corr(all_predictors(), threshold = .9) %>%
        step_log(all_outcomes()) %>%
        check_missing(all_predictors())

model_recipe

## Preparing the recipe
prepped_recipe <- prep(model_recipe, training = hp_train)
prepped_recipe

## Bake the recipe
train <- bake(prepped_recipe, new_data = hp_train)
test <- bake(prepped_recipe, new_data = hp_test)


## Check the data 
anyNA(subset(train, select=-SalePrice))

qqnorm(train$SalePrice)
qqline(train$SalePrice)
#skim(train)

# Modeling

## Determin the best nRound value (number of iterations) with a xgboost model

params<-list(
        max_depth = 4, # default=6
        eta = 0.02, # it lies between 0.01 - 0.3
        gamma = 0, # default=0
        colsample_bytree = 0.65, # Typically, its values lie between (0.5,0.9)
        subsample = 0.6, # Typically, its values lie between (0.5-0.8)
        min_child_weight = 3 # default=1
)

# preparing matrix 
dtrain <- xgb.DMatrix(data = sparse.model.matrix(SalePrice~ .-Id, train), label= train$SalePrice)

# calculate the best nround for this model
set.seed(2123)
xgbcv <- xgb.cv( params = params, data = dtrain, label = train$SalePrice, nrounds = 2000, nfold = 10, showsd = F, stratified = T, print_every_n = 250, early_stopping_rounds = 50, maximize = F)

niter <- xgbcv$best_iteration 

# check the most important features
xgb1 <- xgb.train(data = dtrain, params=params, nrounds = niter)
mat <- xgb.importance(feature_names = xgb1$feature_names, model = xgb1)
ggplot(mat[1:40,])+
        geom_bar(aes(x=reorder(Feature, Gain), y=Gain), stat='identity', fill='red')+
        xlab(label = "Features")+
        coord_flip() +
        ggtitle("Feature Importance")


# Caret ensemble model training

set.seed(2123)
trControl <- trainControl(
        method='cv',
        savePredictions="final",
        index = createFolds(train$SalePrice,k = 10, returnTrain = TRUE),
        allowParallel =TRUE, 
        verboseIter = TRUE
)

xgbGrid <- expand.grid(nrounds = niter, max_depth = c(3,4,5), eta = 0.02, gamma = 0, colsample_bytree = c(0.65),  subsample = c(0.6), min_child_weight = c(3,4,5))
glmGrid <- expand.grid(alpha = 1, lambda = seq(0.00001,0.01,by = 0.0001))
svmGrid <- expand.grid(sigma= 2^seq(-11, -16, -0.5), C= 2^seq(4,9,1))

modelList <<- caretList(
        x = subset(train, select=-c(Id, SalePrice)),
        y = train$SalePrice,
        trControl=trControl,
        metric="RMSE",
        tuneList=list(
                xgbTree = caretModelSpec(method="xgbTree",  tuneGrid = xgbGrid),
                glmnet=caretModelSpec(method="glmnet", tuneGrid = glmGrid),
                svmRadial = caretModelSpec(method="svmRadial", tuneGrid = svmGrid, preProcess=c("nzv", "pca"))
        )
)

#  Model performance

## visualize the tuning result 
## performance of each model 
## correlation between models

plot(modelList$xgbTree)
plot(modelList$glmnet)
plot(modelList$svmRadial)
bwplot(resamples(modelList),metric="RMSE")
modelCor(resamples(modelList))

# Final stacking on the three models

model_Ensemble <- caretEnsemble(   
        modelList, 
        metric="RMSE",
        trControl=trainControl(number=10, method = "repeatedcv", repeats=3)
)
summary(model_Ensemble)


# Check the residuals

pred.train <- predict(model_Ensemble, newdata=subset(train, select=-c(Id, SalePrice)))
hp.plot <- hp %>%
        filter(!is.na(SalePrice))
hp.plot$pred <- pred.train
hp.plot <- hp.plot %>%
        mutate(residual = log(SalePrice)-pred) 



ggplot(hp.plot, aes(x = pred, y = residual)) + 
        geom_pointrange(aes(ymin = 0, ymax = residual)) + 
        geom_hline(yintercept = 0, linetype = 3) + 
        geom_text(data = hp.plot[abs(hp.plot$residual) > 0.4,], aes(label = Id), vjust=1.5, col = "red") +
        ggtitle("Residuals vs. model prediction") +
        xlab("prediction") +
        ylab("residual") +
        theme(text = element_text(size=9)) 


# Final prediction 

finalPredictions <- predict(model_Ensemble, newdata=subset(test, select=-c(Id, SalePrice)))
finalPredictions <- data.frame('Id'= hp_test$Id, 'SalePrice'=exp(finalPredictions)) 
write.csv(finalPredictions, file="finalpredictions.csv", row.names = F)