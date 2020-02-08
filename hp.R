library("tidyverse")
library("caret")
library("DataExplorer")
library("skimr")

setwd("C:/Users/Utilizador/Desktop/kaggle/House_Prices-Advanced_Regression_Techniques")
set.seed(4561)

# Load data
train <- read.csv('train.csv', stringsAsFactors = F)
test <- read.csv('test.csv', stringsAsFactors = F)
glimpse(train)
glimpse(test)
test$SalePrice <- NA
hp <- rbind(train, test)
remove(train); remove(test)
glimpse(hp)
hp %>% skim()

## Fix variable name starts with number

names(hp)[44:45] <- c("FirstFLSF","SecFLSF")
names(hp)[70] <- c("ThreeSnPorch")

# Missing data

apply(sapply(hp, is.na),2,sum)
DataExplorer::plot_missing(hp)

# Feature engineering

## MSSubClass


