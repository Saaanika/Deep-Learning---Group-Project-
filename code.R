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
# ------------------------------------------------------------------------------

# Loading train and test data into dataframes 
train_df <- read.csv("Corona_NLP_train.csv", stringsAsFactors = FALSE)
test_df <- read.csv("Corona_NLP_test.csv",  stringsAsFactors = FALSE)


# Retaining only the variables required for sentiment classification:
# - OriginalTweet: tweet text
# - Sentiment: manually assigned sentiment category
train_texts <- train_df$OriginalTweet
test_texts <- test_df$OriginalTweet
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
# ------------------------------------------------------------------------------

# `maxlen` controls the length of each input sequence fed into the model (since each sequence must be of same length, shorter tweets get padded with 0 and longer tweets may get truncated); need to find max number of words in a tweet 

# Splitting each tweet by whitespace (\\s+) and counting number of words
word_counts <- sapply(train_texts, function(x) length(strsplit(x, "\\s+")[[1]]))   #\\s+ is a regex (splitting on 1 or more whitespace character, eg space or tab)
max(word_counts)   #maximum length of a single tweet is 64, so max_len can be set to this

# Looking at the distribution to understand typical tweet lengths
hist(word_counts, breaks = 50, main = "Distribution of Tweet Lengths (words)", 
     xlab = "Number of Words")
median(word_counts)   #median tweet length is 32 words
quantile(word_counts, c(0.90, 0.95, 0.99))   #90th percentile is 45, 95th is 48, 99th is 52


# `num_words` controls the size of the vocabulary (or how many unique words the model knows); words frequency distribution can be looked at to determine what's a good number

# Combining all tweets into one string, converting to lowercase, and splitting into words to find all words
all_words <- unlist(strsplit(tolower(paste(train_texts, collapse = " ")), " "))

# Finding the frequency of each word
word_freq <- table(all_words)

cat("Total unique words:", length(word_freq))     #total number of unique words is 129648
cat("Words appearing >= 5 times:", sum(word_freq >= 5))   #number of words appearing >= 5 is 13667
cat("Words appearing >= 10 times:", sum(word_freq >= 10))  #number of words appearing >= 10 is 7957

# The dataset contains a large vocabulary (~130k unique words), but many words occur very infrequently and are unlikely to provide meaningful signal for model training.
# Approximately 13k words appear at least 5 times, representing a more informative and reliable subset of the vocabulary (these are words that occur often enough to contribute to learning patterns).
# Based on this, we set num_words to 15,000 which is slightly above the >=5 frequency threshold, to capture most meaningful terms while allowing a small buffer and excluding rare words


# Setting the parameters based on the exploration (using L to store as integer, following the tutorial's RNN building style)
maxlen <- 64L
num_words <- 15000L


# ------------------------------------------------------------------------------
# Part Three: Building and Applying Vectorizer
# ------------------------------------------------------------------------------

vec_int <- layer_text_vectorization(
  max_tokens = num_words,                      #limiting vocabulary to most frequent words
  standardize = "lower_and_strip_punctuation",
  split  = "whitespace",                   #while these are default, still setting them explicitly for clarity   
  output_mode = "int",
  output_sequence_length = maxlen   #done so external padding is not needed anymore (will only pad shorter tweets, no truncation as maxlen is equal to the maximum length of a tweet)
)


# Building vocabulary on only training data (to prevent leakage)
adapt(vec_int, train_texts)

# Looking at the vocabulary (optional, but wanted to for understanding)
vocab <- vec_int$get_vocabulary()
vocab[1:20]   # First 20 tokens in the vocabulary 
cat("Found", length(vocab), "tokens in learned vocabulary (includes OOV at index 1).\n")   #Confirming 15000 tokens including OOV (Out-Of-Vocabulary) or placeholder token used for words not seen during training


# Applying vectorizer to train and test data
x_train <- as.array(vec_int(matrix(train_texts, ncol = 1)))   #ensures texts passed as column vectors (was having issues with the shape during confirmation before)
x_test <- as.array(vec_int(matrix(test_texts, ncol = 1)))

# Confirming the shape of both train and test 
cat("x_train shape:", paste(dim(x_train), collapse = " x "))    #41157 x 64 
cat("x_test shape: ", paste(dim(x_test), collapse = " x "))     #3798 x 64


# ------------------------------------------------------------------------------
# Part Four: Encoding Labels 
# ------------------------------------------------------------------------------

# Defining the 5 ordered sentiment classes (from most negative to most positive)
sentiment_levels <- c("Extremely Negative", "Negative", "Neutral",
                      "Positive", "Extremely Positive")

# Converting text labels to integers 0 to 4 
y_train_int <- as.integer(factor(train_labels_raw, levels = sentiment_levels)) - 1
y_test_int <- as.integer(factor(test_labels_raw,  levels = sentiment_levels)) - 1

# One-hot encode for categorical_crossentropy
num_classes <- 5
y_train <- to_categorical(y_train_int, num_classes = num_classes)
y_test <- to_categorical(y_test_int,  num_classes = num_classes)

# Check class distribution
table(train_labels_raw)
prop.table(table(train_labels_raw))
# classes are somewhat imbalanced: Positive is the largest at ~28%, Extremely Negative is smallest at ~13%


# ------------------------------------------------------------------------------
# Part Five: Shuffling and Splitting Training Data 
# ------------------------------------------------------------------------------

# Setting seed before the shuffle so the entire pipeline (shuffle + val split) is reproducible
set.seed(123)

I <- sample.int(nrow(x_train))
x_train <- x_train[I, ]
y_train <- y_train[I, ]   

# Validation split (80% training, 20% validation)
n <- nrow(x_train)
val_id <- sample(1:n, size = 0.2 * n)

x_val <- x_train[val_id, ]
y_val <- y_train[val_id, ]

x_train_final <- x_train[-val_id, ]
y_train_final <- y_train[-val_id, ]

# Also keeping integer labels for the custom ordinal metric later (need 0-4 class indices, not one-hot)
y_train_final_int <- y_train_int[I][-val_id]
y_val_int <- y_train_int[I][val_id]


# ------------------------------------------------------------------------------
# Part Six: Building Feedforward Neural Network
# ------------------------------------------------------------------------------

# Setting embedding dimension
embedding_dim <- 32  # Smaller embedding better suited for short tweets (480K vs 1.125M embedding params)  

# FF models use layer_flatten() to collapse the embedding output (64 tokens x 32 dims = 2048 features) into a single vector
# this creates a very large input to the dense layers which makes FF prone to overfitting (memorizing training patterns)

# 6.1 - One Hidden Dense Layer 
ff_model1 <- keras_model_sequential() %>%
  layer_embedding(input_dim    = num_words,      #total number of words (max features)
                  output_dim   = embedding_dim) %>%  #embedding dimension
  layer_flatten() %>%
  layer_dense(units = 32, activation = "relu") %>%  #hidden dense layer, values from tutorial 9
  layer_dense(units = 5, activation = "softmax")    #softmax used for multiclass classification problems (5 classes)

# Specify input shape to ensure the model is built, as input_length is deprecated in the embedding layer
ff_model1$build(input_shape = shape(NULL, maxlen))

# Getting model summary 
summary(ff_model1)

# Compiling first FF model
ff_model1 %>% compile(
  optimizer = optimizer_adam(learning_rate = 1e-3),         #adam optimizer, same learning rate across all models
  loss = "categorical_crossentropy",                        #multiclass classification problem
  metrics = c("accuracy")
)

# Fitting the model to the data
ff_model1history <- ff_model1 %>% fit(
  x_train_final, y_train_final,
  epochs = 20,            
  batch_size = 128,       
  validation_data = list(x_val, y_val)      
)


# 6.2 - One Hidden Dense Layer (with Dropout) 
ff_model12 <- keras_model_sequential() %>%
  layer_embedding(input_dim    = num_words,      #total number of words (max features)
                  output_dim   = embedding_dim) %>%  #embedding dimension
  layer_flatten() %>%
  layer_dense(units = 32, activation = "relu") %>%  #hidden dense layer, values from tutorial 9
  layer_dropout(rate = 0.2) %>%                     #adding dropout layer (rate of 20%)
  layer_dense(units = 5, activation = "softmax")    #softmax used for multiclass classification problems (5 classes)

# Specify input shape to ensure the model is built, as input_length is deprecated in the embedding layer
ff_model12$build(input_shape = shape(NULL, maxlen))

# Getting model summary 
summary(ff_model12)

# Compiling first FF model
ff_model12 %>% compile(
  optimizer = optimizer_adam(learning_rate = 1e-3),          
  loss = "categorical_crossentropy",                        #multiclass classification problem
  metrics = c("accuracy")
)

# Fitting the model to the data
ff_model12history <- ff_model12 %>% fit(
  x_train_final, y_train_final,
  epochs = 20,             
  batch_size = 128,        
  validation_data = list(x_val, y_val)      
)


# 6.3 - Two Hidden Dense Layers

ff_model2 <- keras_model_sequential() %>%
  layer_embedding(
    input_dim = num_words,
    output_dim = embedding_dim,
    input_length = maxlen
  ) %>%
  layer_flatten() %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 5, activation = "softmax")

# Specify input shape to ensure the model is built, as input_length is deprecated in the embedding layer
ff_model2$build(input_shape = shape(NULL, maxlen))

# Getting model summary 
summary(ff_model2)

# Compiling second FF model
ff_model2 %>% compile(
  optimizer = optimizer_adam(learning_rate = 1e-3),   
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

# Fitting the model to the data
ff_model2history <- ff_model2 %>% fit(
  x_train_final, y_train_final,
  epochs = 20,             
  batch_size = 128,        
  validation_data = list(x_val, y_val)          
)


# 6.4 - Two Hidden Dense Layers (with Dropout)

ff_model22 <- keras_model_sequential() %>%
  layer_embedding(
    input_dim = num_words,
    output_dim = embedding_dim,
    input_length = maxlen
  ) %>%
  layer_flatten() %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dropout(rate = 0.2) %>%                     
  layer_dense(units = 32, activation = "relu") %>%
  layer_dropout(rate = 0.2) %>%                     #adding dropout layers after each dense layer (rate of 20%)
  layer_dense(units = 5, activation = "softmax")

# Specify input shape to ensure the model is built, as input_length is deprecated in the embedding layer
ff_model22$build(input_shape = shape(NULL, maxlen))

# Getting model summary 
summary(ff_model22)

# Compiling second FF model
ff_model22 %>% compile(
  optimizer = optimizer_adam(learning_rate = 1e-3),   
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

# Fitting the model to the data
ff_model22history <- ff_model22 %>% fit(
  x_train_final, y_train_final,
  epochs = 20,             
  batch_size = 128,        
  validation_data = list(x_val, y_val)   
)


# ------------------------------------------------------------------------------
# Part Seven: Building Recurrent Neural Network
# ------------------------------------------------------------------------------

# SimpleRNN processes the sequence one token at a time, maintaining a hidden state that gets updated at each step
# known to suffer from vanishing/exploding gradients on longer sequences (our tweets go up to 64 tokens), using 32 units to keep the recurrent layer stable
# dropout variants use only regular dropout (not recurrent_dropout, as it caused training collapse for SimpleRNN)

# 7.1 - One Recurrent Layer
rnn_model1 <- keras_model_sequential() %>%
  layer_embedding(input_dim = num_words,             #total number of words (max features)
                  output_dim = embedding_dim) %>%    #embedding dimension
  layer_simple_rnn(units = 32) %>%                   #single recurrent layer, 32 units; return_sequences = FALSE by default (only returns last hidden state)
  layer_dense(units = 5, activation = "softmax")     #softmax used for multiclass classification (5 classes)

# Specify input shape to ensure the model is built
rnn_model1$build(input_shape = shape(NULL, maxlen))

# Getting model summary
summary(rnn_model1)   #RNN layer only adds 2,080 params (32*32 + 32*32 + 32); most params are in the embedding layer

# Compiling first RNN model
rnn_model1 %>% compile(
  optimizer = optimizer_adam(learning_rate = 1e-3),   #adam optimizer, same learning rate across all models
  loss = "categorical_crossentropy",                  #multiclass classification problem
  metrics = c("accuracy")
)

# Fitting the model to the data
rnn_model1history <- rnn_model1 %>% fit(
  x_train_final, y_train_final,
  epochs = 20,                                       #20 epochs, consistent across all models
  batch_size = 128,                                  #batch size of 128, consistent across all models
  validation_data = list(x_val, y_val)
)


# 7.2 - One Recurrent Layer (with Dropout)
rnn_model12 <- keras_model_sequential() %>%
  layer_embedding(input_dim = num_words,
                  output_dim = embedding_dim) %>%
  layer_simple_rnn(units = 32, 
                   dropout = 0.2) %>%                #regular dropout only at 20% (recurrent_dropout caused training collapse for SimpleRNN)
  layer_dense(units = 5, activation = "softmax")

rnn_model12$build(input_shape = shape(NULL, maxlen))
summary(rnn_model12)   #same params as rnn_model1 (dropout doesn't add parameters)

rnn_model12 %>% compile(
  optimizer = optimizer_adam(learning_rate = 1e-3),
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

rnn_model12history <- rnn_model12 %>% fit(
  x_train_final, y_train_final,
  epochs = 20,
  batch_size = 128,
  validation_data = list(x_val, y_val)
)


# 7.3 - Two Recurrent Layers
rnn_model2 <- keras_model_sequential() %>%
  layer_embedding(input_dim = num_words,
                  output_dim = embedding_dim) %>%
  layer_simple_rnn(units = 32, return_sequences = TRUE) %>%   #return_sequences = TRUE so the full sequence is passed to the next RNN layer (not just last element)
  layer_simple_rnn(units = 32) %>%                            #second RNN layer, returns only last hidden state
  layer_dense(units = 5, activation = "softmax")

rnn_model2$build(input_shape = shape(NULL, maxlen))
summary(rnn_model2)   #second RNN layer adds another 2,080 params

rnn_model2 %>% compile(
  optimizer = optimizer_adam(learning_rate = 1e-3),
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

rnn_model2history <- rnn_model2 %>% fit(
  x_train_final, y_train_final,
  epochs = 20,
  batch_size = 128,
  validation_data = list(x_val, y_val)
)


# 7.4 - Two Recurrent Layers (with Dropout)
rnn_model22 <- keras_model_sequential() %>%
  layer_embedding(input_dim = num_words,
                  output_dim = embedding_dim) %>%
  layer_simple_rnn(units = 32, return_sequences = TRUE, 
                   dropout = 0.2) %>%                #dropout on inputs only for first RNN layer
  layer_simple_rnn(units = 32, 
                   dropout = 0.2) %>%                #dropout on inputs only for second RNN layer
  layer_dense(units = 5, activation = "softmax")

rnn_model22$build(input_shape = shape(NULL, maxlen))
summary(rnn_model22)   #same params as rnn_model2 (dropout doesn't add parameters)

rnn_model22 %>% compile(
  optimizer = optimizer_adam(learning_rate = 1e-3),
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

rnn_model22history <- rnn_model22 %>% fit(
  x_train_final, y_train_final,
  epochs = 20,
  batch_size = 128,
  validation_data = list(x_val, y_val)
)


# ------------------------------------------------------------------------------
# Part Eight: Building LSTM Neural Network
# ------------------------------------------------------------------------------

# LSTM uses a gating mechanism (forget gate, input gate, output gate)  that controls what information to keep or discard at each time step
# this solves the vanishing gradient problem that plagued our SimpleRNN models using 64 units instead of 32 because LSTM's gating mechanism makes it stable enough to handle
# the extra capacity, and the added units help it capture more complex patterns in the text
# also using both dropout AND recurrent_dropout here (unlike RNN where recurrent_dropout caused collapse)
# recurrent_dropout regularizes the hidden-to-hidden weight transitions specifically


# 8.1 - One LSTM Layer
lstm_model1 <- keras_model_sequential() %>%
  layer_embedding(input_dim = num_words,             #total number of words (max features)
                  output_dim = embedding_dim) %>%    #embedding dimension
  layer_lstm(units = 64) %>%                         #64 units; LSTM can handle more units than SimpleRNN without instability
  layer_dense(units = 5, activation = "softmax")     #softmax used for multiclass classification (5 classes)

# Specify input shape to ensure the model is built
lstm_model1$build(input_shape = shape(NULL, maxlen))

# Getting model summary
summary(lstm_model1)   #LSTM layer has 24,832 params (4x more than equivalent SimpleRNN because of the 4 gate matrices)

# Compiling first LSTM model
lstm_model1 %>% compile(
  optimizer = optimizer_adam(learning_rate = 1e-3),   #adam optimizer, same learning rate across all models
  loss = "categorical_crossentropy",                  #multiclass classification problem
  metrics = c("accuracy")
)

# Fitting the model to the data
lstm_model1history <- lstm_model1 %>% fit(
  x_train_final, y_train_final,
  epochs = 20,                                       #20 epochs, consistent across all models
  batch_size = 128,                                  #batch size of 128, consistent across all models
  validation_data = list(x_val, y_val)
)


# 8.2 - One LSTM Layer (with Dropout)
lstm_model12 <- keras_model_sequential() %>%
  layer_embedding(input_dim = num_words,
                  output_dim = embedding_dim) %>%
  layer_lstm(units = 64, 
             dropout = 0.2,                          #drops 20% of inputs to the LSTM layer
             recurrent_dropout = 0.2) %>%            #drops 20% of hidden-to-hidden connections (regularizes the recurrent weights specifically)
  layer_dense(units = 5, activation = "softmax")

lstm_model12$build(input_shape = shape(NULL, maxlen))
summary(lstm_model12)   #same params as lstm_model1 (dropout doesn't add parameters)

lstm_model12 %>% compile(
  optimizer = optimizer_adam(learning_rate = 1e-3),
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

lstm_model12history <- lstm_model12 %>% fit(
  x_train_final, y_train_final,
  epochs = 20,
  batch_size = 128,
  validation_data = list(x_val, y_val)
)


# 8.3 - Two LSTM Layers
lstm_model2 <- keras_model_sequential() %>%
  layer_embedding(input_dim = num_words,
                  output_dim = embedding_dim) %>%
  layer_lstm(units = 64, return_sequences = TRUE) %>%   #return_sequences = TRUE so full sequence goes to next LSTM layer
  layer_lstm(units = 64) %>%                            #second LSTM layer, only returns final hidden state
  layer_dense(units = 5, activation = "softmax")

lstm_model2$build(input_shape = shape(NULL, maxlen))
summary(lstm_model2)   #second LSTM layer has 33,024 params (more than first because input is now 64-dim from first LSTM, not 32 from embedding)

lstm_model2 %>% compile(
  optimizer = optimizer_adam(learning_rate = 1e-3),
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

lstm_model2history <- lstm_model2 %>% fit(
  x_train_final, y_train_final,
  epochs = 20,
  batch_size = 128,
  validation_data = list(x_val, y_val)
)


# 8.4 - Two LSTM Layers (with Dropout)
lstm_model22 <- keras_model_sequential() %>%
  layer_embedding(input_dim = num_words,
                  output_dim = embedding_dim) %>%
  layer_lstm(units = 64, return_sequences = TRUE, 
             dropout = 0.2, recurrent_dropout = 0.2) %>%   #both dropout types on first LSTM
  layer_lstm(units = 64, 
             dropout = 0.2, recurrent_dropout = 0.2) %>%   #both dropout types on second LSTM
  layer_dense(units = 5, activation = "softmax")

lstm_model22$build(input_shape = shape(NULL, maxlen))
summary(lstm_model22)   #same params as lstm_model2 (dropout doesn't add parameters)

lstm_model22 %>% compile(
  optimizer = optimizer_adam(learning_rate = 1e-3),
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

lstm_model22history <- lstm_model22 %>% fit(
  x_train_final, y_train_final,
  epochs = 20,
  batch_size = 128,
  validation_data = list(x_val, y_val)
)

# ------------------------------------------------------------------------------
# Part Nine: Model Comparison
# ------------------------------------------------------------------------------

# putting all the history objects and names into lists so its easier to loop through
all_histories <- list(
  ff_model1history, ff_model12history, ff_model2history, ff_model22history,
  rnn_model1history, rnn_model12history, rnn_model2history, rnn_model22history,
  lstm_model1history, lstm_model12history, lstm_model2history, lstm_model22history
)

model_names <- c(
  "FF 1-layer", "FF 1-layer+dropout", "FF 2-layer", "FF 2-layer+dropout",
  "RNN 1-layer", "RNN 1-layer+dropout", "RNN 2-layer", "RNN 2-layer+dropout",
  "LSTM 1-layer", "LSTM 1-layer+dropout", "LSTM 2-layer", "LSTM 2-layer+dropout"
)

# Getting the best val accuracy across all epochs for each model, not just the final epoch
# Some models overfit so their val accuracy peaks early and then drops
best_val_acc <- sapply(all_histories, function(h) max(h$metrics$val_accuracy))
names(best_val_acc) <- model_names

# Also getting the final epoch val accuracy to see how much it dropped from the peak
final_val_acc <- sapply(all_histories, function(h) tail(h$metrics$val_accuracy, 1))
names(final_val_acc) <- model_names

# Printing val accuracy for each model (best across all epochs vs final epoch)
cat("\nValidation Accuracy Summary:\n")
for (i in seq_along(model_names)) {
  cat(model_names[i], "- Best:", round(best_val_acc[i], 3), 
      " Final:", round(final_val_acc[i], 3), "\n")
}

# Looking at which model did best
best_idx <- which.max(best_val_acc)
cat("\nBest model:", model_names[best_idx], round(best_val_acc[best_idx], 3), "\n")

# ------------------------------------------------------------------------------
# Part Ten: Training History Plots
# ------------------------------------------------------------------------------

# Plotting training curves in 2x2 grids for each model type to show training vs validation accuracy and loss over epochs
# Useful for seeing overfitting

# FF models
par(mfrow = c(2, 2))
plot(ff_model1history, main = "FF 1-layer")
plot(ff_model12history, main = "FF 1-layer + Dropout")
plot(ff_model2history, main = "FF 2-layer")
plot(ff_model22history, main = "FF 2-layer + Dropout")
par(mfrow = c(1, 1))

# RNN models
par(mfrow = c(2, 2))
plot(rnn_model1history, main = "RNN 1-layer")
plot(rnn_model12history, main = "RNN 1-layer + Dropout")
plot(rnn_model2history, main = "RNN 2-layer")
plot(rnn_model22history, main = "RNN 2-layer + Dropout")
par(mfrow = c(1, 1))

# LSTM models
par(mfrow = c(2, 2))
plot(lstm_model1history, main = "LSTM 1-layer")
plot(lstm_model12history, main = "LSTM 1-layer + Dropout")
plot(lstm_model2history, main = "LSTM 2-layer")
plot(lstm_model22history, main = "LSTM 2-layer + Dropout")
par(mfrow = c(1, 1))


# ------------------------------------------------------------------------------
# Part Eleven: Ranked Probability Score (RPS) Evaluation
# ------------------------------------------------------------------------------

# RPS is a scoring rule for ordinal/multicategorical probabilistic forecasts; it compares the predicted CDF against the observed CDF at each threshold
# RPS = (1/(K-1)) * sum_{k=1}^{K-1} (CDF_pred[k] - CDF_true[k])^2
# Key properties:
#   - Uses the FULL predicted probability distribution 
#   - Distance-sensitive: predicting "Extremely Positive" when truth is 
#     "Extremely Negative" is penalized far more than being off by one class
#   - Lower is better where 0 means perfect prediction
#   - Respects the ordinal structure: EN < N < Neu < P < EP


# Creating a function to calculate the RPS score
rps_score <- function(y_true_onehot, y_pred_probs) {     #y_true: one-hot matrix (n_samples x 5)
                                                         #y_pred: softmax probability matrix (n_samples x 5)
  K <- ncol(y_true_onehot)    #number of classes (5)
  
  # Cumulative probabilities for predicted and true labels
  cdf_pred <- t(apply(y_pred_probs, 1, cumsum))
  cdf_true <- t(apply(y_true_onehot, 1, cumsum))
  
  # Computing RPS for each observation
  rps_each <- rowSums((cdf_pred[, 1:(K - 1)] - cdf_true[, 1:(K - 1)])^2) / (K - 1)   #last column excluded for both as CDF is 1 (no contribution); squaring penalizes larger errors more 
  
  # Returning mean RPS across all observations
  return(mean(rps_each))
}



# Applying RPS to all 12 models on the validation set

all_models <- list(
  ff_model1, ff_model12, ff_model2, ff_model22,
  rnn_model1, rnn_model12, rnn_model2, rnn_model22,
  lstm_model1, lstm_model12, lstm_model2, lstm_model22
)


# Computing RPS and accuracy for each model on validation set (for side-by-side comparison)

rps_results <- numeric(length(all_models))    

for (i in seq_along(all_models)) {
  
  # Predicted probabilities from softmax output
  pred_probs <- predict(all_models[[i]], x_val)
  
  # Converting probabilities to predicted class indices (0–4)
  pred_class <- apply(pred_probs, 1, which.max) - 1
  true_class <- apply(y_val, 1, which.max) - 1
  
  # Computing metrics
  rps_results[i] <- rps_score(y_val, pred_probs)
}

# Creating a clean results table 
results <- data.frame(
  Model = model_names,
  RPS = round(rps_results, 4)
)

print(results)

# Identifying best models in terms of RPS and accuracy
best_rps_idx <- which.min(rps_results)


cat("Best model by RPS:", model_names[best_rps_idx],
    "-", round(rps_results[best_rps_idx], 4), "\n")

