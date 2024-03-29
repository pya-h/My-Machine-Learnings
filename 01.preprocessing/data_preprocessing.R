
# constants
DATASET_FILENAME = 'Data.csv'
WORKING_DIR = "~/pya.h/machine-learning/excersizes/01-preprocessing"


# first, let's set the current address as the working directory
#setwd(WORKING_DIR)

# now, read the Data set
Dataset = read.csv(DATASET_FILENAME)
View(Dataset)

# handle the missing values
Dataset$Age = ifelse(test = is.na(Dataset$Age), 
                     yes = ave(Dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)),
                     no = Dataset$Age)
Dataset$Salary = ifelse(test = is.na(Dataset$Salary), 
                        yes = ave(Dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)),
                        n= Dataset$Salary)

# categorizing the non numeric data
Dataset$Country = factor(Dataset$Country,
                         levels=c('France', 'Spain', 'Germany'),
                         labels=c(1,2,3))
Dataset$Purchased = factor(x = Dataset$Purchased, levels=c('Yes', 'No'), labels=c(1, 0))

# split data into training set and test set; program will learn on the training-set and will check its algorithms on the test set
# install.packages('caTools')  # package for splitting 
library('caTools')
Splitter = sample.split(Dataset$Purchased, SplitRatio = 3/4) # Splitter will be an array of TRUE/FALSE variables
# this vector will be used as a splitting sample
TrainingSet = subset(Dataset, Splitter)  # or Splitter == TRUE => subset will get a second parameter that is a vector 
# of Boolean variables and selects the true one
TestSet = subset(Dataset, Splitter == FALSE)

TrainingSet[, 2:3] = scale(TrainingSet[, 2:3])
TestSet[, 2:3] = scale(TestSet[, 2:3])
View(TrainingSet)
View(TestSet)