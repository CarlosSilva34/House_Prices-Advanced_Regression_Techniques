library("tidyverse")
library("caret")
library("data.table")
library("gridExtra")
library("skimr")
library("recipes")

setwd("C:/Users/Utilizador/Desktop/kaggle/House_Prices-Advanced_Regression_Techniques")
set.seed(4561)

# Load data
train <- read.csv('train.csv')
test <- read.csv('test.csv')
glimpse(train)
glimpse(test)
test$SalePrice <- NA
hp <- rbind(train, test)
remove(train); remove(test)
glimpse(hp)
skim(hp)

# Changing MSSubClass to a factor

hp <- hp %>%
        mutate(MSSubClass = factor(MSSubClass))

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
        geom_text(data=hp[GrLivArea>4500,], mapping=aes(label=Id), vjust=1.5, col = "blue") +
        xlab("above grade living area") +
        ylab("sale price")


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
               totalPorchSF = OpenPorchSF+EnclosedPorch+ThreeSnPorch+ScreenPorch,
               withbsmtBath = as.numeric(BsmtHalfBath + BsmtFullBath!=0),
               isBsmtUnf = as.numeric(TotalBsmtSF == BsmtUnfSF)
        )

# Data preprocessing - Use recipe to do preprocessing

## split the data
hp_train <- hp[!is.na(SalePrice),]
hp_test <- hp[is.na(SalePrice),]
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

summary(model_recipe)

## Preparing the recipe
prepped_recipe <- prep(model_recipe, training = hp_train)
prepped_recipe

## Bake the recipe
train <- bake(prepped_recipe, new_data = hp_train)
test <- bake(prepped_recipe, new_data = hp_test)


#Check the data 
anyNA(subset(train, select=-SalePrice))

qqnorm(train$SalePrice)


# Modeling