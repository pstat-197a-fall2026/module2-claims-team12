# Primary task (a): binary classification -- two further experiments
# 1. Combine our two winning NN tweaks (higher dropout + early stopping)
# 2. Try a completely different method: SVM on TF-IDF features
#
# Same train/test split as all earlier scripts (same seed, same data)
# so every number stays comparable.

library(tidyverse)
library(tidymodels)
library(tidytext)
library(textstem)
library(keras3)
library(tensorflow)
library(e1071)

load('data/claims-clean-binary.RData')   # produced by binary-baseline.R

set.seed(110122)
partitions <- claims_clean %>%
  initial_split(prop = 0.8)

train_df <- training(partitions)
test_df  <- testing(partitions)

class_metrics <- metric_set(sensitivity, specificity, accuracy, roc_auc)

## -----------------------------------------------------------------
## EXPERIMENT 1: combo NN (dropout 0.5 + early stopping, 20 epochs)
## -----------------------------------------------------------------

train_text <- train_df %>% pull(text_clean)
train_labels <- train_df %>% pull(bclass) %>% as.numeric() - 1
test_text <- test_df %>% pull(text_clean)
test_labels <- test_df %>% pull(bclass) %>% as.numeric() - 1

preprocess_layer <- layer_text_vectorization(
  standardize = NULL, split = 'whitespace', ngrams = NULL,
  max_tokens = NULL, output_mode = 'tf_idf'
)
preprocess_layer %>% adapt(train_text)

combo_model <- keras_model_sequential() %>%
  preprocess_layer() %>%
  layer_dropout(0.5) %>%
  layer_dense(units = 25) %>%
  layer_dropout(0.5) %>%
  layer_dense(1) %>%
  layer_activation(activation = 'sigmoid')

combo_model %>% compile(
  loss = 'binary_crossentropy', optimizer = 'adam', metrics = 'binary_accuracy'
)

combo_model %>% fit(
  train_text, train_labels,
  validation_split = 0.3, epochs = 20,
  callbacks = list(callback_early_stopping(
    monitor = 'val_loss', patience = 2, restore_best_weights = TRUE
  )),
  verbose = 0
)

combo_probs <- predict(combo_model, test_text, verbose = 0) %>% as.numeric()
combo_pred <- as.numeric(combo_probs > 0.5)

combo_results <- tibble(
  truth = factor(test_labels, levels = c(0, 1), labels = c('N/A', 'Relevant')),
  pred  = factor(combo_pred, levels = c(0, 1), labels = c('N/A', 'Relevant')),
  prob  = combo_probs
) %>%
  class_metrics(truth = truth, estimate = pred, prob, event_level = 'second') %>%
  mutate(variant = 'combo NN (dropout 0.5 + early stopping)')

combo_results

## -----------------------------------------------------------------
## EXPERIMENT 2: SVM on TF-IDF features
## -----------------------------------------------------------------
## Vocabulary and word-importance weights (idf) are computed from
## TRAINING data only, then applied to the test set -- same principle
## as adapt() for the neural net, to avoid the test set leaking into
## feature construction.

tokenize_count <- function(df) {
  df %>%
    unnest_tokens(output = token, input = text_clean, token = 'words',
                  stopwords = str_remove_all(stop_words$word, '[[:punct:]]')) %>%
    mutate(token.lem = lemmatize_words(token)) %>%
    filter(str_length(token.lem) > 2) %>%
    count(.id, token.lem, name = 'n')
}

train_tokens <- tokenize_count(train_df)
test_tokens  <- tokenize_count(test_df)

# idf (word importance weights) computed from training documents only
train_tfidf <- train_tokens %>%
  bind_tf_idf(term = token.lem, document = .id, n = n)

idf_lookup <- train_tfidf %>% distinct(token.lem, idf)

# restrict to the 2000 most common training words, for a manageable
# and reasonably fast feature matrix
vocab <- train_tokens %>%
  count(token.lem, wt = n, sort = TRUE) %>%
  slice_head(n = 2000) %>%
  pull(token.lem)

build_features <- function(tokens, vocab, idf_lookup) {
  tokens %>%
    filter(token.lem %in% vocab) %>%
    group_by(.id) %>%
    mutate(tf = n / sum(n)) %>%
    ungroup() %>%
    left_join(idf_lookup, by = 'token.lem') %>%
    mutate(tf_idf = tf * idf) %>%
    pivot_wider(id_cols = .id, names_from = token.lem,
                values_from = tf_idf, values_fill = 0) %>%
    # ensure every vocab column exists, even if unused in this split
    { bind_cols(., as_tibble(matrix(0, nrow(.), sum(!vocab %in% names(.)),
                                     dimnames = list(NULL, setdiff(vocab, names(.)))))) }
}

train_features <- build_features(train_tokens, vocab, idf_lookup)
test_features  <- build_features(test_tokens, vocab, idf_lookup)

# attach labels, keep only rows with computed features, align column order
train_svm_df <- train_df %>%
  select(.id, bclass) %>%
  inner_join(train_features, by = '.id') %>%
  select(-.id)

test_svm_df <- test_df %>%
  select(.id, bclass) %>%
  inner_join(test_features, by = '.id') %>%
  select(-.id)

# match column order between train/test (defensive, in case join reorders)
common_cols <- union('bclass', vocab)
train_svm_df <- train_svm_df %>% select(all_of(intersect(common_cols, names(train_svm_df))))
test_svm_df  <- test_svm_df  %>% select(all_of(names(train_svm_df)))

svm_fit <- svm(bclass ~ ., data = train_svm_df, kernel = 'linear', probability = TRUE)

svm_pred_raw <- predict(svm_fit, newdata = test_svm_df, probability = TRUE)
svm_probs <- attr(svm_pred_raw, 'probabilities')[, 'Relevant claim content']

svm_results <- tibble(
  truth = test_svm_df$bclass,
  pred  = svm_pred_raw,
  prob  = svm_probs
) %>%
  class_metrics(truth = truth, estimate = pred, prob, event_level = 'second') %>%
  mutate(variant = 'SVM (linear kernel, TF-IDF top 2000 words)')

svm_results

## -----------------------------------------------------------------
## FULL COMPARISON so far
## -----------------------------------------------------------------

prior_results <- readRDS('results/binary_tuning_comparison.rds') %>%
  pivot_longer(-variant, names_to = '.metric', values_to = '.estimate')

all_results <- bind_rows(prior_results, combo_results, svm_results)

final_comparison <- all_results %>%
  select(variant, .metric, .estimate) %>%
  pivot_wider(names_from = .metric, values_from = .estimate) %>%
  arrange(desc(accuracy))

final_comparison

saveRDS(final_comparison, 'results/binary_advanced_comparison.rds')
