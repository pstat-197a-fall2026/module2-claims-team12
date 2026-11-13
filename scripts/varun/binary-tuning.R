# Primary task (a): binary classification -- tuning the baseline network
#
# The baseline (scripts/varun/binary-baseline.R) showed signs of
# overfitting: training accuracy climbed to ~92% while validation
# accuracy stalled around 76-82%. This script tries several simple
# adjustments, one at a time, to see which actually helps on held-out
# test data -- not just on training data.

library(tidyverse)
library(tidymodels)
library(keras3)
library(tensorflow)

load('data/claims-clean-binary.RData')   # produced by binary-baseline.R

set.seed(110122)
partitions <- claims_clean %>%
  initial_split(prop = 0.8)

train_text <- training(partitions) %>% pull(text_clean)
train_labels <- training(partitions) %>% pull(bclass) %>% as.numeric() - 1
test_text <- testing(partitions) %>% pull(text_clean)
test_labels <- testing(partitions) %>% pull(bclass) %>% as.numeric() - 1

class_metrics <- metric_set(sensitivity, specificity, accuracy, roc_auc)

# One function that builds + trains + evaluates a network, so every
# variant below is tested in exactly the same way -- only the one
# setting we're curious about changes each time.
run_variant <- function(label, max_tokens = NULL, dense_units = 25,
                         dropout_rate = 0.2, epochs = 5,
                         early_stopping = FALSE) {

  preprocess_layer <- layer_text_vectorization(
    standardize = NULL,
    split = 'whitespace',
    ngrams = NULL,
    max_tokens = max_tokens,
    output_mode = 'tf_idf'
  )
  preprocess_layer %>% adapt(train_text)

  model <- keras_model_sequential() %>%
    preprocess_layer() %>%
    layer_dropout(dropout_rate) %>%
    layer_dense(units = dense_units) %>%
    layer_dropout(dropout_rate) %>%
    layer_dense(1) %>%
    layer_activation(activation = 'sigmoid')

  model %>% compile(
    loss = 'binary_crossentropy',
    optimizer = 'adam',
    metrics = 'binary_accuracy'
  )

  callbacks <- list()
  if (early_stopping) {
    callbacks <- list(callback_early_stopping(
      monitor = 'val_loss', patience = 2, restore_best_weights = TRUE
    ))
  }

  model %>% fit(
    train_text, train_labels,
    validation_split = 0.3,
    epochs = epochs,
    callbacks = callbacks,
    verbose = 0
  )

  test_probs <- predict(model, test_text, verbose = 0) %>% as.numeric()
  test_pred <- as.numeric(test_probs > 0.5)

  tibble(
    truth = factor(test_labels, levels = c(0, 1), labels = c('N/A', 'Relevant')),
    pred  = factor(test_pred,   levels = c(0, 1), labels = c('N/A', 'Relevant')),
    prob  = test_probs
  ) %>%
    class_metrics(truth = truth, estimate = pred, prob, event_level = 'second') %>%
    mutate(variant = label)
}

## Run each variant -- same data, same test set, only one setting changes
############################################################################

results <- bind_rows(
  run_variant('1. baseline (25 units, dropout 0.2, 5 epochs)'),
  run_variant('2. more epochs + early stopping',
              epochs = 20, early_stopping = TRUE),
  run_variant('3. higher dropout (0.5)',
              dropout_rate = 0.5, epochs = 20, early_stopping = TRUE),
  run_variant('4. smaller vocabulary (max 3000 words)',
              max_tokens = 3000, epochs = 20, early_stopping = TRUE),
  run_variant('5. bigger network (64 units)',
              dense_units = 64, epochs = 20, early_stopping = TRUE)
)

comparison <- results %>%
  select(variant, .metric, .estimate) %>%
  pivot_wider(names_from = .metric, values_from = .estimate) %>%
  arrange(desc(accuracy))

comparison

saveRDS(comparison, 'results/binary_tuning_comparison.rds')
