# Group Project - Deep Learning
# Team 2: Avery, Krisha, Andy, Shervin, Saanika 

# ------------------------------------------------------------------------------

# Introduction: COVID-19 Tweet Sentiment Classification
# This project uses a dataset of tweets related to Coronavirus, where each tweet has been manually labeled into one of five sentiment categories
# The categories range from extremely negative to extremely positive.
# The objective of this project is to build, train, and evaluate 3 deep learning models to classify tweets into the correct sentiment category.
# The 3 deep learning models will be:  Feedforward Neural Network (FF), Recurrent Neural Network (RNN), and Long Short-Term Memory (LSTM)


# Loading Relevant Modules
library(keras3)


# ------------------------------------------------------------------------------

# Part One: Loading Data 

# Loading train and test data into dataframes 
train_df <- read.csv("Corona_NLP_train.csv", stringsAsFactors = FALSE)
test_df  <- read.csv("Corona_NLP_test.csv",  stringsAsFactors = FALSE)


# Retaining only the variables required for sentiment classification:
# - OriginalTweet: tweet text
# - Sentiment: manually assigned sentiment category
train_texts <- train_df$OriginalTweet
test_texts  <- test_df$OriginalTweet
train_labels_raw <- train_df$Sentiment
test_labels_raw  <- test_df$Sentiment

# Cleaning text to ensure consistent encoding (with no special symbols); had to be done as was having errors with processing
train_texts <- iconv(train_texts, to = "UTF-8", sub = "")
test_texts  <- iconv(test_texts,  to = "UTF-8", sub = "")


# Verifying number of samples and counts
summary(train_texts)      
summary(train_labels_raw)     #41157 training samples (no missing labels)

summary(test_texts)
summary(test_labels_raw)      #3798 test samples (no missing labels)


# ------------------------------------------------------------------------------

# Part Two: Exploring to Define Parameters

# `maxlen` controls the length of each input sequence fed into the model (since each sequence must be of same length, shorter tweets get padded with 0 and longer tweets may get truncated); need to find max number of words in a tweet 

# Splitting each tweet by whitespace (\\s+) and counting number of words
word_counts <- sapply(train_texts, function(x) length(strsplit(x, "\\s+")[[1]]))   #\\s+ is a regex (splitting on 1 or more whitespace character, eg space or tab)
max(word_counts)   #maximum length of a single tweet is 64, so max_len can be set to this


# `num_words` controls the size of the vocabulary (or how many unique words the model knows);words frequency distribution can be looked at to determine what's a good number

# Combining all tweets into one string, converting to lowercase, and splitting into words to find all words
all_words <- unlist(strsplit(tolower(paste(train_texts, collapse = " ")), " "))

# Finding the frequency of each word
word_freq <- table(all_words)


cat("Total unique words:", length(word_freq))     #total number of unique words is 129648
cat("Words appearing >= 5 times:", sum(word_freq >= 5))   #number of words appearing >= 5 is 13667
cat("Words appearing >= 10 times:", sum(word_freq >= 10))  #number of words appearing >= 5 is 7957

# The dataset contains a large vocabulary (~130k unique words), but many words occur very infrequently and are unlikely to provide meaningful signal for model training.
# Approximately 13k words appear at least 5 times, representing a more informative and reliable subset of the vocabulary (these are words that occur often enough to contribute to learning patterns).
# Based on this, we set num_words to 15,000 which is slightly above the ≥5 frequency threshold, to capture most meaningful terms while allowing a small buffer and excluding rare words


# Setting the parameters based on the exploration (using L to store as integer, following the tutorial's RNN building style)
maxlen <- 64L
num_words <- 15000L


# ------------------------------------------------------------------------------

# Part Three: Building and Applying Vectorizer

vec_int <- layer_text_vectorization(
  max_tokens  = num_words,                      #limiting vocabulary to most frequent words
  standardize = "lower_and_strip_punctuation",
  split       = "whitespace",                   #while these are default, still setting them explicitly for clarity   
  output_mode = "int",
  output_sequence_length = maxlen   #done so external padding is not needed anymore  (will only pad shorter tweets, no truncation as maxlen is equal to the maximum length of a tweet)
)


# Building vocabulary on only training data (to prevent leakage)
adapt(vec_int, train_texts)

# Looking at the vocabulary (optional, bu wanted to for understanding)
vocab <- vec_int$get_vocabulary()
vocab[1:20]   # First 20 tokens in the vocabulary 
cat("Found", length(vocab), "tokens in learned vocabulary (includes OOV at index 1).\n")   #Confirming 15000 tokens including OOV (Out-Of-Vocabulary) or placeholder token used for words not seen during training


# Applying vectorizer to train and test data
x_train <- as.array(vec_int(matrix(train_texts, ncol = 1)))   #ensures tests passed as column vectors (was having issues with the shape during confirmation before)
x_test  <- as.array(vec_int(matrix(test_texts, ncol = 1)))

# Confirming the shape of both train and test 
cat("x_train shape:", paste(dim(x_train), collapse = " x "))    #41157 x 64 
cat("x_test shape: ", paste(dim(x_test), collapse = " x "))     #3798 x 64


# ------------------------------------------------------------------------------

# Part Four: Encoding Labels 

# Defining the 5 ordered sentiment classes (from most negative to most positive)
sentiment_levels <- c("Extremely Negative", "Negative", "Neutral",
                      "Positive", "Extremely Positive")


# Converting text labels to integers 0 to 4 
y_train_int <- as.integer(factor(train_labels_raw, levels = sentiment_levels)) - 1
y_test_int  <- as.integer(factor(test_labels_raw,  levels = sentiment_levels)) - 1

# One-hot encode for categorical_crossentropy
num_classes <- 5
y_train <- to_categorical(y_train_int, num_classes = num_classes)
y_test  <- to_categorical(y_test_int,  num_classes = num_classes)

# Check class distribution
table(train_labels_raw)
prop.table(table(train_labels_raw))

# ------------------------------------------------------------------------------

# Part Five: Shuffling and Splitting Training Data 
I <- sample.int(nrow(x_train))
x_train     <- x_train[I, ]
y_train     <- y_train[I, ]   


# Train and Test Validation Split
set.seed(123)

n <- nrow(x_train)
val_id <- sample(1:n, size = 0.2 * n)

x_val <- x_train[val_id, ]
y_val <- y_train[val_id, ]

x_train_final <- x_train[-val_id, ]
y_train_final <- y_train[-val_id, ]

# ------------------------------------------------------------------------------

# Part Five: Building Feedforward Neural Network

# Setting embedding dimension
embedding_dim <- 75  #as 100 was used for 20,000 word vocabulary from tutorial, 75 was chosen (proportional) 


# 5.1 - One Hidden Dense Layer (with Embedding) 
ff_model1 <- keras_model_sequential() %>%
  layer_embedding(input_dim    = num_words,      #total number of words (max features)
                  output_dim   = embedding_dim) %>%  #embedding dimension
  layer_flatten() %>%
  layer_dense(units = 32, activation = "relu") %>%  #hidden dense layer, values from tutorial 9
  layer_dense(units = 5, activation = "softmax")    #softmax used for multiclass classification problems (5 classes)


# Specify input shape to ensure the model is built, as input_length is deprecated in the embedding layer
ff_model1$build(input_shape = shape(NULL, maxlen))

# Getting model summary 
summary(ff_model1)   #total and trainable parameters: 1,278,797

# Compiling first FF model
ff_model1 %>% compile(
  optimizer = optimizer_rmsprop(learning_rate = 1e-4),      #rmsprop commonly used (keeping learning rate value low)
  loss = "categorical_crossentropy",                        #multiclass classification problem
  metrics = c("accuracy")
)


# Fitting the model to the data
ff_model1history <- ff_model1 %>% fit(
  x_train_final, y_train_final,
  epochs = 15,             #15 epochs was working the best
  batch_size = 32,        #values taken from tutorial (will be kept consistent for all models for fair comparison), 
  validation_data = list(x_val, y_val)   
)

