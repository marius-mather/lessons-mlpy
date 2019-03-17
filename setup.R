list_of_pkgs <- c(
"AmesHousing", 
"GGally", 
"Rtsne", 
"bestNormalize", 
"caret", 
"cluster", 
"corrplot", 
"data.table", 
"dendextend" , 
"e1071", 
"factoextra", 
"gbm", 
"ggplot2", 
"kernlab", 
"lattice", 
"mlbench", 
"naniar" , 
"pROC", 
"pheatmap" , 
"plotROC", 
"psych", 
"ranger", 
"tidymodels", 
"tidyverse", 
"xgboost", 
"vip")

# run the following line of code to install the packages you currently do not have
new_pkgs <- list_of_pkgs[!(list_of_pkgs %in% installed.packages()[,"Package"])]
if(length(new_pkgs)) install.packages(new_pkgs)
