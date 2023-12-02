# load libraries
if (!require(haven))
  install.packages("haven", repos = "http://cran.us.r-project.org")
if (!require(dplyr))
  install.packages("dplyr", repos = "http://cran.us.r-project.org")
if (!require(GGally))
  install.packages("GGally", repos = "http://cran.us.r-project.org")

library(haven)
library(dplyr)
library(GGally)

# Load XPT files
demo <- read_xpt('./datasets/P_DEMO.XPT')
cholest <- read_xpt('./datasets/P_TCHOL.XPT')
body <- read_xpt('./datasets/P_BMX.XPT')
blood <- read_xpt('./datasets/P_PBCD.XPT')

# merge datasets into one
data <- merge(demo, cholest, 'SEQN')
data <- merge(data, body, 'SEQN')
data <- merge(data, blood, 'SEQN')

# interested feature list
cols <- c('LBXTC', 'BMXWT', 'BMXBMI', 'LBXBPB',
          'LBXBCD', 'LBXTHG', 'LBXBSE', 'LBXBMN',
          'RIDAGEYR', 'INDFMPIR', 'RIAGENDR')
data <- data[,cols]

# check for missing values
check_missing_data <- function(data) {
  # Cholesterol Data
  print(paste('Missing cholesterol:', sum(is.na(data$LBXTC))))
  
  # Body Measure Data
  print(paste('Missing weight:', sum(is.na(data$BMXWT))))
  print(paste('Missing body mass index:', sum(is.na(data$BMXBMI))))
  
  # Blood Data
  print(paste('Missing lead concentration:', sum(is.na(data$LBXBPB))))
  print(paste('Missing cadmium concentration:', sum(is.na(data$LBXBCD))))
  print(paste('Missing mercury concentration:', sum(is.na(data$LBXTHG))))
  print(paste('Missing selenium concentration:', sum(is.na(data$LBXBSE))))
  print(paste('Missing manganese concentration:', sum(is.na(data$LBXBMN))))
  
  # Demographic Data
  print(paste('Missing gender:', sum(is.na(data$RIAGENDR))))
  print(paste('Missing age:', sum(is.na(data$RIDAGEYR))))
  print(paste('Missing income status:', sum(is.na(data$INDFMPIR))))
  print(paste('Total data:', nrow(data)))
  print('')
}

# remove missing values & check result
check_missing_data(data)
data <- data[-which(is.na(data$INDFMPIR)),]
data <- data[-which(is.na(data$LBXTC)),]
data <- data[-which(is.na(data$BMXBMI)),]
data <- data[-which(is.na(data$LBXBPB)),]
check_missing_data(data)

# save resulting data
data$LBXTC <- as.factor(ifelse(data$LBXTC > 200, 1, 0))
data$RIAGENDR <- as.factor(data$RIAGENDR)
write.csv(data, './datasets/data.csv', row.names = FALSE)

# create and save partitioned data
data_young <- filter(data, RIDAGEYR < 30)
data_mid <- filter(data, RIDAGEYR >= 30 & RIDAGEYR < 60)
data_old <- filter(data, RIDAGEYR >= 60)
write.csv(data_young, './datasets/data_young.csv', row.names = FALSE)
write.csv(data_mid, './datasets/data_mid.csv', row.names = FALSE)
write.csv(data_old, './datasets/data_old.csv', row.names = FALSE)

# print summary of datasets
print_data_summary <- function(dataset, name) {
  healthy <- round(mean(dataset$LBXTC == 0), 2)
  unhealthy <- round(mean(dataset$LBXTC == 1), 2)
  print(paste('----------', name, '----------'))
  print(paste('Healthy proprtion:', healthy))
  print(paste('Unhealthy proprtion:', unhealthy))
}
print_data_summary(data, 'All People')
print_data_summary(data_young, 'Young People')
print_data_summary(data_mid, 'Middle Age People')
print_data_summary(data_old, 'Elderly People')

fig <- ggpairs(data[,2:10], axisLabels = "none", ggplot2::aes(colour=data$LBXTC), 
        upper = list(continuous = wrap("points", size=0.001), combo = "dot"),
        lower = list(continuous = wrap("cor", size = 2), combo = "dot"),
        diag = list("densityDiag")) + theme_bw(base_size = 6.4)
ggsave('./figures/data_visual.png', plot = fig)

table(data$LBXTC, data$RIAGENDR)