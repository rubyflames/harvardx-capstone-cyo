##########################################################
# Create core set (for training), validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(recommenderlab)) install.packages("recommenderlab", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(recommenderlab)

# ================================================
# Extracting data from the files
# ================================================

# Steam Store Games (Clean dataset), gathered around May 2019:
# https://www.kaggle.com/nikdavis/steam-store-games
# https://www.kaggle.com/nikdavis/steam-store-games?select=steam.csv 

# Steam Video Games dataset ("Recommend video games from 200k interactions user interactions.", gathered around 2017):
# https://www.kaggle.com/tamber/steam-video-games/download

# `catalogue` dataset derived from `interactions.csv` from the Steam Store Games (Clean dataset) folder
catalogue <- fread("interactions.csv",
                 col.names = c("appid",	"name",	"release_date",	"english",
                               "developer",	"publisher",	"platforms",	
                               "required_age",	"categories",	"genres",	
                               "steamspy_tags",	"achievements",	"positive_ratings",	
                               "negative_ratings",	"average_playtime", 
                               "median_playtime", "owners",	"price"))

# `interactions` dataset derived from `steam-200k.csv` file i.e. Steam Video Games dataset that displays 200k Steam user interactions
interactions <- fread("steam-200k.csv",
               col.names = c("userid", "Name", "Purchase_play", "Hours", "Dummy"))

# Check for NA values in both datasets
anyNA(catalogue) # [1] FALSE
anyNA(interactions) # [1] FALSE

# ================================================
# Interactions data - Analysis
# ================================================
summary(interactions)

dim(interactions)

# Since Steam is a digital distribution service for video games, digital copies of games have to be purchased before playing.

nrow(interactions[interactions$Purchase_play == "play"])
# [1] 70489
nrow(interactions[interactions$Purchase_play == "purchase"])
# [1] 129511
# Notice that there are users who had purchased games and not played them on Steam

count(interactions[interactions$Hours == 1.0])
# n: 130569
# Also, based on count alone, not all `Hours` values that are 1.0 refer to purchases


# ================================================
# Catalogue data - Analysis
# ================================================

# First 6 rows of Catalogue dataset
head(catalogue)

# Catalogue dataset summary:
summary(catalogue)

dim(catalogue)
# [1] 27075    18

# Checking distinct values for columns that matter:
catalogue %>% summarise(n_appid = n_distinct(appid), n_name = n_distinct(name), n_publisher = n_distinct(publisher), n_genres = n_distinct(genres), n_tags = n_distinct(steamspy_tags))
# n_appid n_name n_publisher n_genres n_tags
#   27075  27033       14261     1552   6423

# Since there are less unique name values than appid values, we check to see how many games have duplicated names with different ids
n_occur <- data.frame(table(catalogue$name))
# gives you a dataframe with a list of names and the number of times they occurred.

catalogue_duplicates <- n_occur[n_occur$Freq > 1,]
catalogue_duplicates # names which occurred more than once

# Since both datasets would be ultimately joined together, we will check to see if the duplicated names appear in the Interactions dataset
sum(catalogue_duplicates$Var1 %in% interactions$Name)
# 6 of the duplicated titles are included in the `interactions` dataset

# Hence to avoid confusion, we would not include the `appid` column in the combined dataset.

# Notice that there are multiple terms used in the `genres` column separated by `;`:
all_genres <- catalogue %>% separate_rows(genres, sep = "\\;") %>%
  select(genres) %>% unique() # get unique genres
# sort in alphabetical order
all_genres <- all_genres[order(all_genres),]

dim(all_genres) # [1] 29  1
#There are 29 unique genres in this dataset

# ================================================
# Cleaning Catalogue data
# ================================================

# For this analysis, we would consider the following columns from Catalogue dataset:
# name, release_date, developer, publisher, genres, positive_ratings, negative_ratings, price 

catalogue_clean <- catalogue %>% select(name, release_date, developer, publisher, genres, positive_ratings, negative_ratings, price)
summary(catalogue_clean)

# We will add another column called `user_ratings`.
# This would show the rounded-up percentage of positive ratings over total ratings (positive ratings + negative ratings)
catalogue_clean <- catalogue_clean %>%
  mutate(user_ratings = ceiling(positive_ratings*100/(positive_ratings + negative_ratings)) ) %>% 
  select(-positive_ratings, -negative_ratings) # Remove positive_ratings and negative_ratings columns 

# We will create a `year` column from the `release_date`, then remove the `release_date` column
catalogue_clean <- catalogue_clean %>%
  mutate(year = as.numeric(year(release_date))) %>% # create `year` column
  select(-release_date) # remove `release_date`


# ================================================
# Cleaning Interactions data
# ================================================

# Since the last column `Dummy` only has 0s, remove dummy column
interactions <- interactions %>% select(-Dummy)

# Create an `hours_played` column based on `Purchase_play` values and `Hours`
interactions_clean <- interactions %>% mutate(hours_played = case_when(
  endsWith(Purchase_play, "play") ~ Hours,
  endsWith(Purchase_play, "purchase") ~ 0
))
# Then, we aggregate the `interactions_clean` dataset by summing up Hours played according to User ID and game title (Name).
# This reduces confusion on the `Hours` column and removes the need for the `Purchase_play` column.
interactions_clean <- aggregate(hours_played~userid+Name, data=interactions_clean, FUN=sum) 

# We also change all the column names to lowercase make it easier when we join the datasets later.
colnames(interactions_clean) <- c("userid", "name", "hours_played")


# Analysing `interactions_clean` dataset
interactions_clean %>% summarise(n_users = n_distinct(userid), n_games = n_distinct(name))
#   n_users n_games
#1   12393    5155

# ==============================================================================
# Creating the Combined dataset from Interactions and Catalogue datasets
# ==============================================================================

# Perform an inner join on `interactions_clean` and `catalogue_clean` datasets
# This ensures that only complete rows in this combined dataset would be used to train, test and validate the model.
combined <- interactions_clean %>% inner_join(catalogue_clean, by = "name")

# ================================================
# Analysing the Combined dataset
# ================================================
summary(combined)

combined %>% summarise(n_users = n_distinct(userid),
                       n_games = n_distinct(name),
                       n_genres = n_distinct(genres),
                       n_publishers = n_distinct(publisher),
                       n_developers = n_distinct(developer),
                       hours_played_min = min(hours_played), 
                       hours_played_max = max(hours_played),
                       release_min = min(year),
                       release_max = max(year),
                       price_min = min(price),
                       price_max = max(price))
#  n_users n_games n_genres n_publishers n_developers
#   10122    2191      310         1250         1690


# Plot user ids against number of games purchased/played in `combined` dataset
combined %>%
  dplyr::count(userid) %>% 
  ggplot(aes(n)) + 
  geom_histogram(fill = "orange", color = "black") + 
  scale_y_log10() +
  ggtitle("Users vs number of games purchased/played")

# Top 20 users by number of games purchased/played 
combined %>% #filter(is.na(userid) == FALSE) %>% # exclude rows with NA as values
  group_by(userid) %>%
  summarise(count = n()) %>%
  arrange(desc(count)) %>% top_n(20)


# In fact, by looking at the graph below, we can see that relatively few games had users who played for more than 1000 hours

combined %>%
  select(hours_played, name) %>% # each row represents a game played by a single user
  ggplot(aes(name, hours_played)) + 
  geom_point(alpha = 0.2) + 
  ggtitle("Name (of game) vs hours_played")


# Summary of Hours Played
combined %>% select(hours_played) %>% summary()

# In fact, based on the summary, the median (i.e. 50th percentile) was 0.9 hours played, while the 3rd quartile (75th percentile) was 7.8 hours played.
# However, the mean was 34.9 hours played, which is more than 4 times that of the 3rd quartile.
# What is even more surprising is that both the minimum and the 1st quartile (25th percentile) were 0.0 hours played (i.e. purchased but not played at all).
# So let's analyse these observations even further.

combined %>%
  group_by(hours_played) %>%
  summarise(count = n(), pct_of_total = 100*(count/nrow(combined)) ) %>%
  arrange(desc(count)) %>% top_n(20)

# By sorting the number and percentage of occurences of `hours_played` values in descending order,
# we saw that 35.7% of the time, 0.0 hours were played for games purchased on Steam.
# This means that games are hoarded and not played about 35% of the time.

# Summarising occurrences of Hours played that are above the mean of 34.9 hours
above_avg_hours_played <- combined %>% group_by(hours_played) %>%
  summarise(count = n(), pct_of_total = 100*(count/nrow(combined)) ) %>%
  arrange(desc(count)) %>% filter(hours_played>=34.9)

sum(above_avg_hours_played$pct_of_total)
# [1] 10.59525

# Around 10.6% of the time, users in the combined dataset played Steam games above the average number of hours.

# Hence, recommendation engines should recommend games that the user would likely purchase and play 

# User Ratings vs Hours played
combined %>% 
  ggplot(aes(user_ratings, hours_played)) + 
  geom_point(alpha = 0.2) +
  ggtitle("Average User Ratings vs Hours played")

# From the chart shown, there is generally a normally distributed trend between User Ratings to Hours played, with the most number of hours played peaking at 82


# Top played games by total number of hours played
combined %>% select(name, hours_played) %>% group_by(name) %>% summarise(total_hours_played = sum(hours_played)) %>% 
  top_n(10) %>% arrange(desc(total_hours_played))

# Unique genres in the combined dataset
combined %>% group_by(genres) %>% mutate(count=n()) %>%
  ggplot(aes(genres, count)) + 
  geom_line() +
  ggtitle("Genres vs Number of Game-User combinations")

# ================================================
# Create validate, train and test data sets
# ================================================

## Create data partition to create validation data set:

set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
validation_index <- createDataPartition(y = combined$hours_played, times = 1, p = 0.1, list = FALSE)

# Data set for training and testing model
combined_model <- combined[-validation_index,]

# Validation data set
combined_validate <- combined[validation_index,]  

# Creating Test and Train sets
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = combined_model$hours_played, times = 1, p = 0.1, list = FALSE)
# Test set
test_set <- combined_model[test_index,]

# Train Set
train_set <- combined_model[-test_index,]

# To ensure that we are not testing on games and users we have not seen before
test_set <- test_set %>% semi_join(train_set, by='name') %>%
  semi_join(train_set, by='userid')
#rm(test_index)

## Calculating RMSE

# Defining the RMSE function used in this project
RMSE <- function(true_hours = NULL, predicted_hours = NULL) {
  sqrt(mean((true_hours - predicted_hours)^2))
}

# ================================================
# Training Models
# ================================================

#######################################################################################################
# MODEL 0 - Naive RMSE - apply basic model with using just the average as a prediction
#######################################################################################################

# This model predicts the average hours played for all games-user combinations
mu <- train_set$hours_played %>% mean()
naive_rmse <- RMSE(test_set$hours_played, mu)

# To ensure we do not test on games/users we have never seen before:
#test_set <- test_set %>% semi_join(train_set, by='Name') %>% semi_join(train_set, by='userid')
# rm(test_index)
methods <- ('Model 0 - Just the average')
rmses <- (naive_rmse)

model_evaluations <- tibble(method=methods, RMSE=rmses)


#######################################################################################################
# MODEL 1 - + Name Bias - builds on previous by introducing a Name bias term
#######################################################################################################

name_avgs <- train_set %>% group_by(name) %>% summarise(b_i=mean(hours_played-mu))
name_bias_model <- mu + test_set %>% left_join(name_avgs, by='name') %>% pull(b_i)
name_bias_rmse <- RMSE(test_set$hours_played, name_bias_model)

methods <- c('Model 0 - Just the average', 'Model 1 - added Name bias')
rmses <- c(naive_rmse, name_bias_rmse)
(model_evaluations <- tibble(method=methods, RMSE=rmses))

#######################################################################################################
# MODEL 2 - + User Bias - builds on previous by introducing a User bias term
#######################################################################################################

user_avgs <- train_set %>% left_join(name_avgs, by='name') %>% group_by(userid) %>% summarise(b_u=mean(hours_played-mu-b_i))

user_name_bias_model <- test_set %>% left_join(name_avgs, by='name') %>% left_join(user_avgs, by='userid') %>%
  mutate(pred=mu+b_i+b_u) %>% pull(pred)

user_name_bias_rmse <- RMSE(test_set$hours_played, user_name_bias_model)

methods <- c('Model 0 - Just the average', 'Model 1 - added Name bias', 
             'Model 2 - added User bias')
rmses <- c(naive_rmse, name_bias_rmse, user_name_bias_rmse)

(model_evaluations <- tibble(method=methods, RMSE=rmses))
# Notice that the added User bias makes the RMSE even higher than the Naive RMSE (Model 0)


#######################################################################################################
# Model 3 - + Genre Bias - builds on previous by introducing a Genre bias term
#######################################################################################################

genre_avgs <- train_set %>% left_join(name_avgs, by='name') %>% left_join(user_avgs, by='userid') %>% 
  group_by(genres) %>% summarise(b_g=mean(hours_played-mu-b_i-b_u))
genre_user_name_bias_model <- test_set %>% left_join(name_avgs, by='name') %>% left_join(user_avgs, by='userid') %>%
  left_join(genre_avgs, by='genres') %>% mutate(pred=mu+b_i+b_u+b_g) %>% pull(pred)
(genre_user_name_bias_rmse <- RMSE(test_set$hours_played, genre_user_name_bias_model))
#[1] 235.3351

methods <- c('Model 0 - Just the average', 'Model 1 - added Name bias', 'Model 2 - added User bias', 
             'Model 3 - added Genre bias')
rmses <- c(naive_rmse, name_bias_rmse, user_name_bias_rmse, 
           genre_user_name_bias_rmse)

(model_evaluations <- tibble(method=methods, RMSE=rmses))

#######################################################################################################
# Model 4 - Model 3 + User Rating Bias - builds on previous by introducing a User Rating bias term
#######################################################################################################

rating_avgs <- train_set %>% left_join(name_avgs, by='name') %>% left_join(user_avgs, by='userid') %>% 
  left_join(genre_avgs, by='genres') %>% group_by(user_ratings) %>% summarise(b_r=mean(hours_played-mu-b_i-b_u-b_g))
rating_genre_user_name_bias_model <- test_set %>% left_join(name_avgs, by='name') %>% left_join(user_avgs, by='userid') %>%
  left_join(genre_avgs, by='genres') %>% left_join(rating_avgs, by='user_ratings') %>% mutate(pred=mu+b_i+b_u+b_g+b_r) %>% pull(pred)
(rating_genre_user_name_bias_rmse <- RMSE(test_set$hours_played, rating_genre_user_name_bias_model))
# [1] 235.3351

methods <- c('Model 0 - Just the average', 'Model 1 - added Name bias', 'Model 2 - added User bias', 
             'Model 3 - added Genre bias', 'Model 4 - added User Rating bias')
rmses <- c(naive_rmse, name_bias_rmse, user_name_bias_rmse, 
           genre_user_name_bias_rmse, rating_genre_user_name_bias_rmse)

(model_evaluations <- tibble(method=methods, RMSE=rmses))

#######################################################################################################
# Model 5 - Model 3 + Price Bias - builds on Model 3 by introducing a Price bias term
#######################################################################################################
price_avgs <- train_set %>% left_join(name_avgs, by='name') %>% left_join(user_avgs, by='userid') %>% 
  left_join(genre_avgs, by='genres') %>% group_by(price) %>% summarise(b_p=mean(hours_played-mu-b_i-b_u-b_g))
price_genre_user_name_bias_model <- test_set %>% left_join(name_avgs, by='name') %>% left_join(user_avgs, by='userid') %>%
  left_join(genre_avgs, by='genres') %>% left_join(price_avgs, by='price') %>% mutate(pred=mu+b_i+b_u+b_g+b_p) %>% pull(pred)
(price_genre_user_name_bias_rmse <- RMSE(test_set$hours_played, price_genre_user_name_bias_model))
# [1] 235.3351

methods <- c('Model 0 - Just the average', 'Model 1 - added Name bias', 'Model 2 - added User bias', 
             'Model 3 - added Genre bias', 'Model 4 - added User Rating bias', 
             'Model 5 - added Price bias to Model 3')
rmses <- c(naive_rmse, name_bias_rmse, user_name_bias_rmse, 
           genre_user_name_bias_rmse, rating_genre_user_name_bias_rmse, 
           price_genre_user_name_bias_rmse)

(model_evaluations <- tibble(method=methods, RMSE=rmses))
#Since the RMSE is the same regardless of factoring additional bias on top of Model 3 (Added Name, User and Genre) , we can conclude that other factors do not impact the hours played by users.

################################################
# Model 6 - Reversed Bias Terms
###############################################
# This model builds on mu by introducing a user bias term
user_avgs <- train_set %>% group_by(userid) %>% summarise(b_u=mean(hours_played-mu))
user_bias_model <- mu + test_set %>% left_join(user_avgs, by='userid') %>% pull(b_u)
(rb_user_bias_rmse <- RMSE(test_set$hours_played, user_bias_model))

# This model builds on previous by introducing a name bias term
name_avgs <- train_set %>% left_join(user_avgs, by='userid') %>% group_by(name) %>% summarise(b_i=mean(hours_played-mu-b_u))
name_user_bias_model <- test_set %>% left_join(name_avgs, by='name') %>% left_join(user_avgs, by='userid') %>%
  mutate(pred=mu+b_i+b_u) %>% pull(pred)
(rb_name_user_bias_rmse <- RMSE(test_set$hours_played, name_user_bias_model))

methods <- c('Model 0 - Just the average', 'Model 1 - added Name bias', 'Model 2 - added User bias', 
             'Model 3 - added Genre bias', 'Model 4 - added User Rating bias', 
             'Model 5 - added Price bias to Model 3', 'Model 6a - Reverse Bias - added User bias', 
             'Model 6b - Reverse Bias - added Name bias')
rmses <- c(naive_rmse, name_bias_rmse, user_name_bias_rmse, 
           genre_user_name_bias_rmse, rating_genre_user_name_bias_rmse, 
           price_genre_user_name_bias_rmse, rb_user_bias_rmse, 
           rb_name_user_bias_rmse)

(model_evaluations <- tibble(method=methods, RMSE=rmses))

################################################
# Model 7 - Regularisation & Cross Validation
################################################

# This model builds on biased model by introducing a regularisation (lambda) term 

# Use cross-validation to search for best lambda term:
lambdas <- seq(0, 70, 1)
rmses <- sapply(lambdas, function(l){
  mu <- mean(train_set$hours_played)
  b_i <- train_set %>% group_by(name) %>% summarise(b_i=sum(hours_played-mu)/(n()+l))
  
  b_u <- train_set %>% left_join(b_i, by='name') %>%
    group_by(userid) %>% summarise(b_u=sum(hours_played-b_i-mu)/(n()+l))
  
  predicted_hours <- test_set %>% left_join(b_i, by='name') %>%
    left_join(b_u, by='userid') %>%
    mutate(b_i = coalesce(b_i, 0), b_u = coalesce(b_u, 0), pred=mu+b_i+b_u) %>% # replace missing values with 0 for b_i & b_u
    pull(pred)
  
  return(RMSE(predicted_hours, test_set$hours_played))
})
# visualise the search for best lambda
qplot(lambdas, rmses)
# save the best lambda term
(lambda <- lambdas[which.min(rmses)])
#[1] 44

#Regularised name bias term
b_i <- train_set %>% group_by(name) %>% summarise(b_i=sum(hours_played-mu)/(n()+lambda))
#Regularised user bias term
b_u <- train_set %>% left_join(b_i, by='name') %>%
  group_by(userid) %>% summarise(b_u=sum(hours_played-b_i-mu)/(n()+lambda))

regularised_user_name_model <- test_set %>% 
  left_join(b_i, by='name') %>%
  left_join(b_u, by='userid') %>% 
  mutate(b_i = coalesce(b_i, 0), b_u = coalesce(b_u, 0), pred=mu+b_i+b_u) %>% # replace missing values with 0 for b_i & b_u
  pull(pred)
regularised_user_name_rmse <- RMSE(test_set$hours_played, regularised_user_name_model)

methods <- c('Model 0 - Just the average', 'Model 1 - added Name bias', 'Model 2 - added User bias', 
             'Model 3 - added Genre bias', 'Model 4 - added User Rating bias', 
             'Model 5 - added Price bias to Model 3', 'Model 6a - Reverse Bias - added User bias', 
             'Model 6b - Reverse Bias - added Name bias', 'Model 7 - Regularised model')
rmses <- c(naive_rmse, name_bias_rmse, user_name_bias_rmse, 
           genre_user_name_bias_rmse, rating_genre_user_name_bias_rmse, 
           price_genre_user_name_bias_rmse, rb_user_bias_rmse, 
           rb_name_user_bias_rmse, regularised_user_name_rmse)

(model_evaluations <- tibble(method=methods, RMSE=rmses))

# Since we notice that the rounded up RMSEs of Model 1 (Average with added Name bias) & Model 7 (Regularised Model) are the same,
# we shall check to see which is smaller:
(name_bias_rmse) # [1] 213.1429
(regularised_user_name_rmse) # [1] 212.896
rmse_check <- c(name_bias_rmse, regularised_user_name_rmse)
min(rmse_check) # [1] 212.896
# Since the Regularised Model has the smaller RMSE, we will go with Model 7 for the final validation.

##########################################################
# Final Validation - Regularisation & Cross Validation
##########################################################

# We will use the Regularisation and Cross Validation model for validation.
# So we will train the model with the combined_model set and test with the combined_validation dataset.

# Use cross-validation to search for best lambda term:
lambdas <- seq(0, 70, 1)
rmses <- sapply(lambdas, function(l){
  mu <- mean(combined_model$hours_played)
  b_i <- combined_model %>% group_by(name) %>% summarise(b_i=sum(hours_played-mu)/(n()+l))
  
  b_u <- combined_model %>% left_join(b_i, by='name') %>%
    group_by(userid) %>% summarise(b_u=sum(hours_played-b_i-mu)/(n()+l))
  
  predicted_hours <- combined_validate %>% left_join(b_i, by='name') %>%
    left_join(b_u, by='userid') %>%
    mutate(b_i = coalesce(b_i, 0), b_u = coalesce(b_u, 0), pred=mu+b_i+b_u) %>% # replace missing values with 0 for b_i & b_u
    pull(pred)
  
  return(RMSE(predicted_hours, combined_validate$hours_played))
})
# visualise the search for best lambda
qplot(lambdas, rmses)
# save the best lambda term
(lambda <- lambdas[which.min(rmses)])
#[1] 65

# Regularised name bias term
b_i <- combined_model %>% group_by(name) %>% summarise(b_i=sum(hours_played-mu)/(n()+lambda))
#Regularised user bias term
b_u <- combined_model %>% left_join(b_i, by='name') %>%
  group_by(userid) %>% summarise(b_u=sum(hours_played-b_i-mu)/(n()+lambda))

# Calculating final model on validation set:

final_model <- combined_validate %>% 
  left_join(b_i, by='name') %>%
  left_join(b_u, by='userid') %>% 
  mutate(b_i = coalesce(b_i, 0), b_u = coalesce(b_u, 0), pred=mu+b_i+b_u) %>% # replace missing values with 0 for b_i & b_u
  pull(pred)
(final_rmse <- RMSE(combined_validate$hours_played, final_model))

methods <- c('Model 0 - Just the average', 'Model 1 - added Name bias', 'Model 2 - added User bias', 
             'Model 3 - added Genre bias', 'Model 4 - added User Rating bias', 
             'Model 5 - added Price bias to Model 3', 'Model 6a - Reverse Bias - added User bias', 
             'Model 6b - Reverse Bias - added Name bias', 'Model 7 - Regularised model', 'Final Model - Regularised Model on Validation Set')
rmses <- c(naive_rmse, name_bias_rmse, user_name_bias_rmse, 
           genre_user_name_bias_rmse, rating_genre_user_name_bias_rmse, 
           price_genre_user_name_bias_rmse, rb_user_bias_rmse, 
           rb_name_user_bias_rmse, regularised_user_name_rmse, final_rmse)

(model_evaluations <- tibble(method=methods, RMSE=rmses))

# The final model produced an RMSE of approximately 179 hours played, which is lower than that produced by training the model.

