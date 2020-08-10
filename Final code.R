### Code provided for this Capstone Project ####

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)



### Necessary packages ####

# Before proceeding, please install and load the packages below :

if(!require(dendextend)) install.packages("dendextend", repos = "http://cran.us.r-project.org")
if(!require(polycor)) install.packages("polycor", repos = "http://cran.us.r-project.org")
if(!require(psych)) install.packages("psych", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(ggcorrplot)) install.packages("ggcorrplot", repos = "http://cran.us.r-project.org")
if(!require(missMDA)) install.packages("missMDA", repos = "http://cran.us.r-project.org")
if(!require(GPArotation)) install.packages("GPArotation", repos = "http://cran.us.r-project.org")
if(!require(FactoMineR)) install.packages("FactoMineR", repos = "http://cran.us.r-project.org")
if(!require(factoextra)) install.packages("factoextra", repos = "http://cran.us.r-project.org")
if(!require(recommenderlab)) install.packages("recommenderlab", repos = "http://cran.us.r-project.org")
if(!require(recosystem)) install.packages("recosystem", repos = "http://cran.us.r-project.org")
if(!require(microbenchmark)) install.packages("microbenchmark", repos = "http://cran.us.r-project.org")



### Structure of Data ####
str(edx)
head(edx)


### Data exploration ####
summary(edx)

edx %>% summarize(users = n_distinct(userId),
                titles = n_distinct(title),
                movid = n_distinct(movieId),
                genres = n_distinct(genres),
                mu = mean(edx$rating))

### Distribution of ratings among users #####

edx %>% group_by(userId) %>% summarize(n = n()) %>% arrange(desc(n))

# Plot
edx %>% group_by(userId) %>% summarize(n = n()) %>%
  ggplot(aes(n))+
  geom_histogram(bins = 30)+
  ggtitle("Distribution of ratings among users")+
  xlab("Users")


### Distribution of ratings among movies #####

edx %>% group_by(title) %>% summarize(n = n()) %>% arrange(desc(n))

# Plot
edx %>% group_by(title) %>% summarize(n = n()) %>%
  ggplot(aes(n))+
  geom_histogram(bins=30)+
  ggtitle("Number of rated movies")+
  xlab("Movies")

### Sparsity of data #####

# Let's pick a subset of the first 10,000 observations :
edx_wide <- edx[1:10000,] %>% 
  select(userId, movieId, rating) %>% 
  spread(movieId, rating)

# Image of sparsity
image(as.matrix(edx_wide[,-1]),
      main = "Sparsity of data",
      xlab = "Users",
      ylab = "Movies",
      col = "blue")

# Median of the dataset
median(edx$rating)


### Rating distribution #####

hist(edx$rating, main = "Rating distribution")


### Genres effect ####

edx %>% group_by(genres) %>%
  summarize(n = n(), avg = mean(rating), se = sd(rating)/sqrt(n())) %>%
  filter(n >= 1000) %>% 
  mutate(genres = reorder(genres, avg)) %>%
  ggplot(aes(x = genres, y = avg, ymin = avg - 2*se, ymax = avg + 2*se)) + 
  geom_point() +
  geom_errorbar() + 
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  ggtitle("Genres effect")

### Hierarchical clustering ####

# Creating a dense matrix

top_users <- edx %>% 
  group_by(userId) %>%
  summarize(n=n()) %>%
  arrange(desc(n)) %>%
  top_n(20) %>%
  pull(userId)

top_movies <- edx %>% 
  group_by(movieId) %>%
  summarize(n=n()) %>%
  arrange(desc(n)) %>%
  top_n(20) %>%
  pull(movieId)

top_df <- edx %>%
  filter(movieId %in% top_movies, userId %in% top_users)

top_matrix <- top_df %>%
  select(userId, title, rating) %>%
  spread(title, rating)

# Image of dense matrix

image(as.matrix(top_matrix[,-1]),
      main = "Dense matrix",
      xlab = "Users",
      ylab = "Movies",
      col = "blue")

# Calculating distances

dist_set <- dist(t(top_matrix[,-1]), method = "euclidean")

# Clustering

hc_set <- hclust(dist_set, method = "complete")

# Dendrogram

plot(hc_set, cex = 0.65, main = "", xlab = "")

# Distance between users
dist_users <- dist(top_matrix[,-1], method = "euclidean")

# Clusters of users
hc_users <- hclust(dist_users, method = "complete")

# Dendrogram
plot(hc_users, cex = 0.65, main = "", xlab = "")

# Coloured dendrogram
library(dendextend)

dend <- as.dendrogram(hc_users)
color_dend <- color_branches(dend, h = 5)
plot(color_dend, cex=0.65)

# 9th user's ratings
top_matrix[9,]

# 20th user's ratings
top_matrix[20,]

### Exploratory Factor Analysis ####

library(polycor)

# Getting a correlation matrix
hc_top <- hetcor(top_matrix[,-1], use = "pairwise.complete.obs")

# Getting the correlations
polyc_top <- hc_top$correlations


library(psych)

# Bartlett test : p-value should be < 0.05
cortest.bartlett(polyc_top, n=20)

# KMO : results should be close to 1
KMO(polyc_top)


library(ggcorrplot)

#Visualizing correlations
ggcorrplot(polyc_top, hc.order = TRUE, type = "lower")

# Scree plot
scree(polyc_top)

# Parallel analysis scree plot
fa.parallel(polyc_top, n.obs = 20, fm="minres" , fa = "fa")


library(GPArotation)

# Factor analysis with 7 factors and oblique rotation
f7_top_obli <- fa(polyc_top, nfactors = 7, rotate = "oblimin")

# Path diagram
fa.diagram(f7_top_obli)

### Principal Component Analysis ####

library(FactoMineR)

# PCA

pca_output <- PCA(top_matrix[,-1], ncp = 4)


library(factoextra)

# Dimensions plot by movie

fviz_pca_ind(pca_output, habillage = top_matrix[,-1], addEllipses = TRUE, repel = TRUE)


### Randomly order the dataset #####

set.seed(69,sample.kind = "Rounding")

rows <- sample(nrow(edx))
random_edx <- edx[rows,]

### Splitting in train and test sets ####

# Create Partition
test_index <- createDataPartition(y = random_edx$rating, times = 1, p = 0.8, list = FALSE)

train_set <- random_edx[test_index,]
test_set <- random_edx[-test_index,]


### Matrix Factorization & Recommender ####

library(recosystem)
library(recommenderlab) # We can use this package for the RMSE function.

# Converting train and test sets in 'Recosystem' objects of class "DataSource"

train_reco <- with(train_set,data_memory(user_index = userId,
                                         item_index = movieId,
                                         rating = rating))

test_reco <- with(test_set,  data_memory(user_index = userId, 
                                         item_index = movieId,
                                         rating = rating))



# Object class
class(train_reco)

### Training the model ####

# Simple training model
set.seed(69, sample.kind = "Rounding") # We must set seed because it's a randomized algorithm (ie, stochastic)

rec <- Reco()
rec$train(train_data = train_reco, opts = c(verbose = FALSE))

# Predicting ratings

test_pred <- rec$predict(test_reco, out_memory())

# RMSE

RMSE(test_set$rating, test_pred)

# Computing time

library(microbenchmark)
set.seed(69, sample.kind = "Rounding")

computing_time <- microbenchmark(
  
  rec$train(train_data = train_reco),
  times = 10 # We calculate this 10 times and we'll get a data frame of 10 different computing times
)

plot(computing_time)


### Tuning parameters #### (WARNING ! LONG COMPUTING TIME : AROUND 2 HOURS)


opt = rec$tune(train_reco, opts = list(dim = c(10, 20),
                                       lrate = c(0.01, 0.1),
                                       costp_l1 = c(0, 0.1),
                                       costp_l2 = c(0.01, 0.1),
                                       costq_l1 = c(0, 0.1),
                                       costq_l2 = c(0.01, 0.1)
))


# Best values
opt$min

# Tuning 'dim' parameter :

d <- c(50, 100, 200, 300)

dim_tune <- sapply(d, function(dim){
  rec$train(train_data = train_reco,opts = c(dim = dim,
                                             costp_l2 = 0.01, # According to opt$min, this is the only best value that is not a default value
                                             verbose = FALSE))
  
  test_pred <- rec$predict(test_reco,out_memory())
  
  RMSE(test_set$rating,test_pred)
})

plot(dim_tune)

# Tuning learning rate 'lrate' : (Takes around 20 minutes)

l <- c(0.1,0.2)

learn_tune <- sapply(l, function(learn){
  rec$train(train_data = train_reco,opts = c(lrate= learn,
                                             dim = 200,# Value from the previous test
                                             costp_l2 = 0.01,
                                             verbose = FALSE))
  
  test_pred <- rec$predict(test_reco,out_memory())
  
  RMSE(test_set$rating,test_pred)
})

plot(learn_tune)

# Tuning 'L1 p and q costs' : (Takes around 20 minutes)

# P1 cost :
c <- c(0,0.01)

cp1_tune <- sapply(c, function(cost){
  rec$train(train_data = train_reco,opts = c(dim = 200,
                                             costp_l1 = cost,
                                             costp_l2 = 0.01,
                                             verbose = FALSE))
  
  test_pred <- rec$predict(test_reco,out_memory())
  
  RMSE(test_set$rating,test_pred)
})

# Plot
plot(cp1_tune)

### Final training model #### (Takes around 3 minutes)

rec$train(train_data = train_reco, opts = c(dim = 200,
                                            costp_l1 = 0,
                                            costp_l2=0.01,
                                            costq_l1 = 0,
                                            costq_l2 = 0.1,
                                            lrate = 0.1,
                                            nfold = 10,
                                            niter = 20,
                                            nmf = TRUE,
                                            verbose = FALSE))


### Predicting results on the test set ####

test_pred <- rec$predict(test_reco, out_memory()) ### Takes a few minutes

# Calculating RMSE with the test set

RMSE(test_set$rating, test_pred)

# Best movies in test set
test_set %>% 
  group_by(title) %>%
  summarize(n=n()) %>%
  arrange(desc(n)) %>%
  slice(1:10) %>%
  pull(title)

# Best movies according to our predictions
test_set %>%
  mutate(predictions = test_pred) %>%
  arrange(desc(predictions)) %>%
  slice(1:10) %>%
  pull(title)

# Worst movies in test set
test_set %>% 
  arrange(rating) %>%
  slice(1:10) %>%
  pull(title)

# Worst movies according to our predictions
test_set %>%
  mutate(predictions = test_pred) %>%
  arrange(predictions) %>%
  slice(1:10) %>%
  pull(title)


### Validation ####

# Converting the 'validation' set in 'Recosystem' object

valid_reco <- with(validation, data_memory(user_index = userId,
                                           item_index = movieId,
                                           rating = rating))

# Getting predictions for the validation set :

valid_pred <- rec$predict(valid_reco, out_memory()) ### Takes a few minutes

### FINAL RMSE ####

RMSE(validation$rating, valid_pred)





