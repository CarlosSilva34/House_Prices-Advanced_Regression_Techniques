library("tidyverse")
library("caret")

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

apply(sapply(hp, is.na),2,sum)