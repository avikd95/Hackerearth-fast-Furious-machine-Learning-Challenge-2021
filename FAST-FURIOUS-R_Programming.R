####################################### Hackerearth Fast Furious Machine Learning Problem ###########################################

# Removing garbages :
gc()

# Increasing memory :
memory.limit(9999999999)

# Install Packages :
library(purrr)
library(caret)
library(lubridate)
library(superml)
library(tidyverse)
library(modelr)
library(dplyr)


# Reading Datasets 
train = read.csv(file.choose(new = FALSE),check.names = FALSE)
test = read.csv(file.choose(new = FALSE),check.names = FALSE)

# Variable Type :
str(train)
str(test)

# NA checking For numeric variable :
which(!complete.cases(train))      # gives row indices with missing values
which(!complete.cases(test))          

############################################ Data Wrangling for train data ############################################

# NA checking For character variable :
which(train$Insurance_company == "")    # for blank cells 
which(train$Expiry_date == "")          # for blank cells
which(train$Insurance_company == NA)    # when by default have NA printed on cells
which(train$Expiry_date == NA)          # when by default have NA printed on cells
map(train, ~sum(is.na(.)))              # number of rows with missing values


# Extracting important variables from Expity_Date :
Expiry_Year = year(ymd(train$Expiry_date))     # converting into year
Expiry_Month = month(ymd(train$Expiry_date))   # converting into month
Expiry_Day = day(ymd(train$Expiry_date))       # converting into day


# Converting Category variables :
label <- LabelEncoder$new()
Company = label$fit_transform(train$Insurance_company)


# Unnecessary columns : Image_path 

####### train without Amount column 
train1 = data.frame(Company ,"Cost_of_vehicle" = train$Cost_of_vehicle ,
                    "Min_coverage" = train$Min_coverage , Expiry_Year ,
                    Expiry_Month , Expiry_Day , "Max_coverage" = train$Max_coverage ,
                    "Condition" = as.factor(train$Condition)
                    )

# contains all the columns excluding "Image_path" , "Amount" 

# Missing Value Imputation :
set.seed(50)
train1 = kNN(train1 , k = 38 )[,1:8]    # k = sqrt(number of observations)
apply(train1,2,sd)


# removing unnecessary varibles to avoid ambiguity :
remove(Company)
remove(Expiry_Day)
remove(Expiry_Month)
remove(Expiry_Year)


############################################# Data Wrangling for test data ##########################################


# Extracting important variables from Expity_Date :
Expiry_Year = year(ymd(test$Expiry_date))     # converting into year
Expiry_Month = month(ymd(test$Expiry_date))   # converting into month
Expiry_Day = day(ymd(test$Expiry_date))       # converting into day


# Converting Category variables :
label <- LabelEncoder$new()
Company = label$fit_transform(test$Insurance_company)


# Unnecessary columns : Image_path 


####### t
test1 = data.frame(Company ,"Cost_of_vehicle" = test$Cost_of_vehicle ,
                    "Min_coverage" = test$Min_coverage , Expiry_Year ,
                    Expiry_Month , Expiry_Day , "Max_coverage" = test$Max_coverage 
                    )

# removing unnecessary varibles to avoid ambiguity :
remove(Company)
remove(Expiry_Day)
remove(Expiry_Month)
remove(Expiry_Year)



# Split the train1 data into train and test :
set.seed(100)

training.samples <- sample(1:nrow(train1),size = 0.8 * nrow(train1) , replace = FALSE)

train1_split = train1[sort(training.samples),]
test1_split =  train1[-sort(training.samples),]


set.seed(200)
model1 <- train(
  Condition ~., data = train1_split, method = "xgbTree", na.action = na.pass,
  trControl = trainControl("cv", number = 10) 
)

model1$bestTune

# Note that prediction is done using best tune parameters by default

# Statistic to check accuracy :
confusionMatrix(test1_split$Condition , predict(model1 , test1_split)) 
# Kappa = 1 denoting the ideal case with all are perfectly classified

# Feature Engineering :
varImp(model1)
plot(varImp(model1),main = "Feature Engineering" ,xlab = "Variable Contribution (%)")

# Important variable is only "Max_Coverage"


####### Prediction of "Condition" on test1 ######

Condition = predict(model1 , test1) 


######## storing it on test data i.e "test1"

test1 = data.frame(test1 , Condition)

# Removing "Condition" column to avoid ambiguity :
remove(Condition)


## We have now "train1" and "test1" datasets required to predict the "Amount"   

##################### Predicting Amount Column ###################

train2 = data.frame(train1 , "Amount" = train$Amount)

set.seed(300)
partition = sort(sample(1:nrow(train2),size = 0.8 * nrow(train2) , replace = FALSE))
                                            

train2_split = train2[sort(partition),]
test2_split =  train2[-sort(partition),]

# stratification on train2_split datasets based on condition column

train2_split = stratified(train2_split, c('Condition'), 0.8)  #restructuring the train2_split dataset



#######################################################  Predictive Modelling  ##################################################

set.seed(400)
model2 <- train(Amount ~., data = train2_split, method = "xgbTree", na.action = na.omit,
               trControl = trainControl("cv", number = 20 )  , 
               objective = "reg:squarederror", metric = "Rsquared" 
)

model2$results

model2$bestTune

# Performance :
rsquare(model2,test2_split)     # Ans: 16.42 %



### More Accuracy (using repeated cv mthod) :

set.seed(500)
model <- train(Amount ~., data = train2_split, method = "xgbTree", na.action = na.omit,
               trControl = trainControl("repeatedcv", number = 20 , repeats = 50)  , 
               objective = "reg:squarederror", metric = "Rsquared"
               )                 
### takes 3.5 hour to execute

model$results

model$bestTune

# Performance :
rsquare(model,test2_split)  # Ans : 16.46 %



##################################################  FEATURE ENGINEERING  ###############################################

varImp(model)

# Plot the model 

plot(varImp(model))


############################################ Prediction on the given test data #########################################

Amount = predict(model, test1)


# Imposing Conditions after seeing the behavior of given train data 

for(i in 1:nrow(test1))
{
  if (test1$Condition[i] == 0)
    Amount[i] = 0 
}


###### Generating Output data frame ######
submission = data.frame("Image_path" = test$Image_path , "Condition" = test1$Condition ,
                        Amount )

remove(Amount)

# It's giving 54.82 leaderboard score.



############################################# Exporting in csv format ##########################################

write.csv(submission , "C:/Users/Avik/Desktop/Fast_Furious_Submission.csv" , row.names = FALSE)






#################################################### DEEP LEARNING USING KERAS ################################################


# Feature Engineering from last model suggest : Expiry_Month is not necessary

# Let's see whether Keras find column "Expiry_Month" to be useful

## Packages ##


## Installation 1 : Using without GPU ##

install.packages("keras")
library(keras)

if ("package:tensorflow" %in% search()) { detach("package:tensorflow", unload=TRUE) }
if ("tensorflow" %in% rownames(installed.packages())) { remove.packages("tensorflow") }

install.packages("tensorflow")
library(tensorflow)
install_tensorflow(package_url = 
"https://pypi.python.org/packages/b8/d6/af3d52dd52150ec4a6ceb7788bfeb2f62ecb6aa2d1172211c4db39b349a2/
tensorflow-1.3.0rc0-cp27-cp27mu-manylinux1_x86_64.whl#md5=1cf77a2360ae2e38dd3578618eacc03b" )


## Installation 2 : Using GPU ##

library(keras)
library(tensorflow)
install_tensorflow(version = "gpu")
library(dplyr)



########### Splitted train data converting into matrix form 

splitted_train = na.omit(train2_split)    # Expiry_Month column is not needed from Xgboost

attach(splitted_train)

Company = as.numeric(scale(Company))
Cost_of_vehicle = as.numeric(scale(Cost_of_vehicle))
Min_coverage = as.numeric(scale(Min_coverage))
Expiry_Year = as.numeric(scale(Expiry_Year))
Expiry_Month = as.numeric(scale(Expiry_Month))
Expiry_Day = as.numeric(scale(Expiry_Day))
Max_coverage = as.numeric(scale(Max_coverage))
Condition = as.numeric(as.vector(Condition))
Amount = as.numeric(Amount)

splitted_train = as.matrix(data.frame(Company,Cost_of_vehicle,Min_coverage,
                                      Expiry_Year,Expiry_Month,Expiry_Day,
                                      Max_coverage,Condition,Amount))

remove(Company)
remove(Cost_of_vehicle)
remove(Min_coverage)
remove(Expiry_Year)
remove(Expiry_Month)
remove(Expiry_Day)
remove(Max_coverage)
remove(Condition)
remove(Amount)


########### Splitted test data converted into matrix form 

splitted_test = na.omit(test2_split)    # Expiry_Month column is not needed from Xgboost

attach(splitted_test)

Company = as.numeric(scale(Company))
Cost_of_vehicle = as.numeric(scale(Cost_of_vehicle))
Min_coverage = as.numeric(scale(Min_coverage))
Expiry_Year = as.numeric(scale(Expiry_Year))
Expiry_Month = as.numeric(scale(Expiry_Month))
Expiry_Day = as.numeric(scale(Expiry_Day))
Max_coverage = as.numeric(scale(Max_coverage))
Condition = as.numeric(as.vector(Condition))
Amount = as.numeric(Amount)


splitted_test = as.matrix(data.frame(Company,Cost_of_vehicle,Min_coverage,
                                      Expiry_Year,Expiry_Month,Expiry_Day,
                                      Max_coverage,Condition,Amount))


remove(Company)
remove(Cost_of_vehicle)
remove(Min_coverage)
remove(Expiry_Year)
remove(Expiry_Month)
remove(Expiry_Day)
remove(Max_coverage)
remove(Condition)
remove(Amount)



# Create the keras model :

# Hyperparameter tunning search results helped us to prepare the following keras model :

library(mgcv)

dlmodel <- vector(mode = "list", length = 10)


for( i in seq_along(dlmodel)) {

dlmodel[[i]] <- keras_model_sequential()  %>% 
  layer_dense(units = 16, activation = "relu",input_shape = 8,
              bias_regularizer = regularizer_l1_l2(l1 = 0.001 , l2 = 0.001)) %>%
  layer_dense(units = 2, activation = "relu") %>%
  layer_alpha_dropout(rate = 0.05) %>%
  layer_dense(units = 1, activation = "linear",
              bias_initializer = initializer_random_normal())


dlmodel[[i]]

dlmodel[[i]] %>% 
  compile(
    loss = "mse",
    optimizer = optimizer_adam(lr = 0.001),
    metrics = list("mean_absolute_error"),
    loss_weights = 0.5
    )

dlmodel[[i]]

# Train the model using average blender :


  dlmodel[[i]] %>% fit(
  x = splitted_train[,-9] ,
  y = splitted_train[,9],
  epochs = 1000,
  validation_split = 0.2,
  verbose = 2 ,
  batch_size = 16
  )
}

# Checking performance on validation data :

observed = splitted_test[,9]


for ( i in seq_along(dlmodel)) {

dlmodel[[i]] %>% evaluate(splitted_test[,-9], splitted_test[,9] , verbose = 0 )


prediction = dlmodel[[i]] %>% predict(splitted_test[,-9])


# Metrics :

cat("Correlation on valid data :", cor(prediction ,observed^2),"\n")

}



## correlation R^2 for validation data is on an average  : 0.25


######## Original Test Data to matrix conversion for keras ########

attach(test1)

Company = as.numeric(scale(Company))
Cost_of_vehicle = as.numeric(scale(Cost_of_vehicle))
Min_coverage = as.numeric(scale(Min_coverage))
Expiry_Year = as.numeric(scale(Expiry_Year))
Expiry_Month = as.numeric(scale(Expiry_Month))
Expiry_Day = as.numeric(scale(Expiry_Day))
Max_coverage = as.numeric(scale(Max_coverage))
Condition = as.numeric(as.vector(Condition))


original_test = as.matrix(data.frame(Company,Cost_of_vehicle,Min_coverage,
                                     Expiry_Year,Expiry_Month,Expiry_Day,
                                     Max_coverage,Condition))


remove(Company)
remove(Cost_of_vehicle)
remove(Min_coverage)
remove(Expiry_Year)
remove(Expiry_Month)
remove(Expiry_Day)
remove(Max_coverage)
remove(Condition)


for ( i in seq_along(dlmodel))
{
  
Amount = cummean(dlmodel[[i]] %>% predict(original_test , batch_size = 8192))
 
}


#### Here Amount is the average estimate much reliable


######## Now those condition which are zero , corresponding amount are also zero in train data
### We use that to prepare the prediction of Amount in the test data 

for(i in 1:nrow(test1))
{
  if (test1$Condition[i] == 0)
    Amount[i] = 0 
}


###### Generating Output data frame ######
submission = data.frame("Image_path" = test$Image_path , "Condition" = test1$Condition ,
                        Amount )

remove(Amount)

remove(dlmodel)


bind_rows(head(submission),tail(submission))  # showing head and tail of "submission"

############################################# Exporting in csv format ##########################################

write.csv(submission , "C:/Users/Avik/Desktop/Fast_Furious_Submission.csv" , row.names = FALSE)


# giving 57.66405 leaderboard score , this has gotten me at top 10 in leaderboard 


################################################################### THE END #############################################################





















