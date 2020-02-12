library("tidyverse")
library("caret")
library("DataExplorer")
library("skimr")

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
dplyr::count(hp, RoofMatl)
hp <- hp %>%
        mutate(RoofMatl = fct_collapse(RoofMatl, "TarGrv" = "Tar&Grv"))
dplyr::count(hp, RoofMatl)

dplyr::count(hp, Exterior1st)
hp <- hp %>%
        mutate(Exterior1st = fct_collapse(Exterior1st, "WdSdng" = "Wd Sdng"))

dplyr::count(hp, Exterior2nd)
hp <- hp %>%
        mutate(Exterior2nd = fct_collapse(Exterior2nd, "BrkComm" = "Brk Cmn", "WdSdng" = "Wd Sdng", "WdShng" = "Wd Shng"))

table(hp$GarageYrBlt)

# Fix typo of GarageYrBlt

summary(hp$GarageYrBlt)
hp <- hp %>% 
        mutate(GarageYrBlt = ifelse(GarageYrBlt > 2010, NA, GarageYrBlt))


# remove Outliers




