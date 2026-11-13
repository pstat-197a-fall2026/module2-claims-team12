# Primary task (a): one last experiment -- threshold tuning.
#
# Every model so far used a flat 0.5 cutoff to decide "relevant" vs
# "N/A". But glmnet's ROC AUC (0.873) was notably higher than its
# accuracy (0.786) suggested -- a sign the predicted probabilities are
# well-ranked, just not optimally split at 0.5. This tunes the cutoff
# properly: using held-out validation predictions (not the final test
# set), picking the threshold that maximizes sensitivity + specificity
# (Youden's J), then evaluating that fixed threshold on the test set.

library(tidyverse)
library(tidymodels)
library(tidytext)
library(textstem)
library(glmnet)
library(Matrix)
library(keras3)
library(tensorflow)

class_metrics <- metric_set(sensitivity, specificity, accuracy, roc_auc)

load('data/claims-clean-binary.RData')
set.seed(110122)
partitions <- claims_clean %>% initial_split(prop = 0.8)
train_df <- training(partitions)
test_df  <- testing(partitions)

## -----------------------------------------------------------------
## glmnet: use cross-validated (out-of-fold) training predictions to
## pick the threshold -- genuinely held-out during training, not the
## test set, so this isn't "cheating" by peeking at test data.
## -----------------------------------------------------------------

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

train_idf <- train_tokens %>%
  bind_tf_idf(term = token.lem, document = .id, n = n) %>%
  distinct(token.lem, idf)

vocab <- sort(unique(train_tokens$token.lem))
vocab_index <- setNames(seq_along(vocab), vocab)

build_sparse <- function(tokens, vocab_index, idf_lookup) {
  tokens <- tokens %>%
    filter(token.lem %in% names(vocab_index)) %>%
    group_by(.id) %>%
    mutate(tf = n / sum(n)) %>%
    ungroup() %>%
    left_join(idf_lookup, by = 'token.lem') %>%
    mutate(tf_idf = tf * idf)
  doc_ids <- unique(tokens$.id)
  doc_index <- setNames(seq_along(doc_ids), doc_ids)
  sparseMatrix(i = doc_index[tokens$.id], j = vocab_index[tokens$token.lem],
               x = tokens$tf_idf, dims = c(length(doc_ids), length(vocab_index)),
               dimnames = list(doc_ids, names(vocab_index)))
}

x_train <- build_sparse(train_tokens, vocab_index, train_idf)
x_test  <- build_sparse(test_tokens, vocab_index, train_idf)

y_train <- train_df %>% filter(.id %in% rownames(x_train)) %>%
  arrange(match(.id, rownames(x_train))) %>% pull(bclass)
y_test <- test_df %>% filter(.id %in% rownames(x_test)) %>%
  arrange(match(.id, rownames(x_test))) %>% pull(bclass)

set.seed(110122)
cv_glmnet <- cv.glmnet(x_train, y_train, family = 'binomial', alpha = 0,
                       nfolds = 5, keep = TRUE)   # keep = out-of-fold preds

lambda_min_index <- which(cv_glmnet$lambda == cv_glmnet$lambda.min)
oof_probs <- cv_glmnet$fit.preval[, lambda_min_index]   # held-out-during-CV probs

find_best_threshold <- function(probs, truth, positive_level) {
  truth_bin <- as.numeric(truth == positive_level)
  thresholds <- seq(0.1, 0.9, by = 0.01)
  j_stat <- map_dbl(thresholds, function(t) {
    pred <- as.numeric(probs > t)
    sens <- sum(pred == 1 & truth_bin == 1) / sum(truth_bin == 1)
    spec <- sum(pred == 0 & truth_bin == 0) / sum(truth_bin == 0)
    sens + spec - 1
  })
  thresholds[which.max(j_stat)]
}

positive_level <- 'Relevant claim content'
best_t_glmnet <- find_best_threshold(oof_probs, y_train, positive_level)
cat('glmnet best threshold (from CV, not test set):', best_t_glmnet, '\n')

test_probs_glmnet <- predict(cv_glmnet, newx = x_test, s = 'lambda.min', type = 'response')[, 1]
test_pred_glmnet <- factor(test_probs_glmnet > best_t_glmnet, labels = levels(y_test))

glmnet_tuned_results <- tibble(truth = y_test, pred = test_pred_glmnet, prob = test_probs_glmnet) %>%
  class_metrics(truth = truth, estimate = pred, prob, event_level = 'second') %>%
  mutate(variant = paste0('glmnet ridge, tuned threshold (t=', best_t_glmnet, ')'))

glmnet_tuned_results

## -----------------------------------------------------------------
## NN: explicit train/val split (not keras's internal one) so we have
## real validation predictions to pick a threshold from.
## -----------------------------------------------------------------

set.seed(110122)
inner_split <- train_df %>% initial_split(prop = 0.7)
train_inner <- training(inner_split)
val_inner   <- testing(inner_split)

train_text <- train_inner %>% pull(text_clean)
train_labels <- train_inner %>% pull(bclass) %>% as.numeric() - 1
val_text <- val_inner %>% pull(text_clean)
val_labels <- val_inner %>% pull(bclass) %>% as.numeric() - 1
test_text <- test_df %>% pull(text_clean)
test_labels <- test_df %>% pull(bclass) %>% as.numeric() - 1

preprocess_layer <- layer_text_vectorization(
  standardize = NULL, split = 'whitespace', ngrams = NULL,
  max_tokens = NULL, output_mode = 'tf_idf'
)
preprocess_layer %>% adapt(train_text)

nn_model <- keras_model_sequential() %>%
  preprocess_layer() %>%
  layer_dropout(0.2) %>%
  layer_dense(units = 25) %>%
  layer_dropout(0.2) %>%
  layer_dense(1) %>%
  layer_activation(activation = 'sigmoid')

nn_model %>% compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = 'binary_accuracy')

nn_model %>% fit(
  train_text, train_labels,
  validation_data = list(val_text, val_labels),
  epochs = 20,
  callbacks = list(callback_early_stopping(monitor = 'val_loss', patience = 2, restore_best_weights = TRUE)),
  verbose = 0
)

val_probs <- predict(nn_model, val_text, verbose = 0) %>% as.numeric()
best_t_nn <- find_best_threshold(val_probs, factor(val_labels, levels = c(0,1), labels = levels(y_test)), positive_level)
cat('NN best threshold (from validation, not test set):', best_t_nn, '\n')

test_probs_nn <- predict(nn_model, test_text, verbose = 0) %>% as.numeric()
test_pred_nn <- as.numeric(test_probs_nn > best_t_nn)

nn_tuned_results <- tibble(
  truth = factor(test_labels, levels = c(0, 1), labels = c('N/A', 'Relevant')),
  pred  = factor(test_pred_nn, levels = c(0, 1), labels = c('N/A', 'Relevant')),
  prob  = test_probs_nn
) %>%
  class_metrics(truth = truth, estimate = pred, prob, event_level = 'second') %>%
  mutate(variant = paste0('NN early-stopping, tuned threshold (t=', best_t_nn, ')'))

nn_tuned_results

## -----------------------------------------------------------------
## FULL COMPARISON
## -----------------------------------------------------------------

prior_results <- readRDS('results/binary_round3_comparison.rds') %>%
  pivot_longer(-variant, names_to = '.metric', values_to = '.estimate')

final_comparison <- bind_rows(prior_results, glmnet_tuned_results, nn_tuned_results) %>%
  select(variant, .metric, .estimate) %>%
  pivot_wider(names_from = .metric, values_from = .estimate) %>%
  arrange(desc(accuracy))

final_comparison

saveRDS(final_comparison, 'results/binary_final_comparison.rds')
saveRDS(list(best_t_glmnet = best_t_glmnet, best_t_nn = best_t_nn),
        'results/binary_tuned_thresholds.rds')
