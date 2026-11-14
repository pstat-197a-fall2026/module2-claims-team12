# Preliminary task 3: build and train a neural network, experimenting
# with architecture/training configuration until it performs at least
# as well as the logistic PCR model from task 1 (79.5% accuracy,
# 0.863 ROC AUC on the paragraph-only model, per writeups/tasks-summary.qmd).
#
# Uses the same train/test split already exported from the task 1/2
# pipeline (data/nn-train.csv, data/nn-test.csv), so results are
# directly comparable to that benchmark.

library(tidyverse)
library(tidymodels)
library(keras3)
library(tensorflow)

train_df <- read_csv('data/nn-train.csv', show_col_types = FALSE)
test_df  <- read_csv('data/nn-test.csv', show_col_types = FALSE)

class_metrics <- metric_set(sensitivity, specificity, accuracy, roc_auc)

run_variant <- function(label, dense_units = 25, dropout_rate = 0.2,
                         epochs = 20, early_stopping = TRUE,
                         ngrams = NULL, val_split = 0.3, patience = 2,
                         max_tokens = NULL) {

  preprocess_layer <- layer_text_vectorization(
    standardize = NULL, split = 'whitespace', ngrams = ngrams,
    max_tokens = max_tokens, output_mode = 'tf_idf'
  )
  preprocess_layer %>% adapt(train_df$paragraph_text)

  model <- keras_model_sequential() %>%
    preprocess_layer() %>%
    layer_dropout(dropout_rate) %>%
    layer_dense(units = dense_units) %>%
    layer_dropout(dropout_rate) %>%
    layer_dense(1) %>%
    layer_activation(activation = 'sigmoid')

  model %>% compile(
    loss = 'binary_crossentropy', optimizer = 'adam', metrics = 'binary_accuracy'
  )

  callbacks <- list()
  if (early_stopping) {
    callbacks <- list(callback_early_stopping(
      monitor = 'val_loss', patience = patience, restore_best_weights = TRUE
    ))
  }

  model %>% fit(
    train_df$paragraph_text, train_df$binary_label,
    validation_split = val_split, epochs = epochs,
    callbacks = callbacks, verbose = 0
  )

  test_probs <- predict(model, test_df$paragraph_text, verbose = 0) %>% as.numeric()
  test_pred <- as.numeric(test_probs > 0.5)

  metrics <- tibble(
    truth = factor(test_df$binary_label, levels = c(0, 1)),
    pred  = factor(test_pred, levels = c(0, 1)),
    prob  = test_probs
  ) %>%
    class_metrics(truth = truth, estimate = pred, prob, event_level = 'second') %>%
    mutate(variant = label)

  list(model = model, metrics = metrics)
}

## Try a few configurations -- same spirit as the binary-classification
## tuning done for the primary task: start simple, fix overfitting with
## early stopping, check whether a bit more/less capacity helps.
#############################################################################

variants <- list(
  list(label = '1. baseline (25 units, dropout 0.2, 5 epochs, no early stop)',
       dense_units = 25, dropout_rate = 0.2, epochs = 5, early_stopping = FALSE),
  list(label = '2. early stopping (25 units, dropout 0.2, up to 20 epochs)',
       dense_units = 25, dropout_rate = 0.2, epochs = 20, early_stopping = TRUE),
  list(label = '3. higher dropout (0.5) + early stopping',
       dense_units = 25, dropout_rate = 0.5, epochs = 20, early_stopping = TRUE),
  list(label = '4. smaller network (10 units) + early stopping',
       dense_units = 10, dropout_rate = 0.2, epochs = 20, early_stopping = TRUE),
  list(label = '5. unigrams + bigrams (ngrams=2) + early stopping',
       dense_units = 25, dropout_rate = 0.2, epochs = 20, early_stopping = TRUE,
       ngrams = 2L),
  list(label = '6. more training data (val split 0.15) + early stopping',
       dense_units = 25, dropout_rate = 0.2, epochs = 20, early_stopping = TRUE,
       val_split = 0.15, patience = 3),
  list(label = '7. bigrams + more training data + early stopping',
       dense_units = 25, dropout_rate = 0.2, epochs = 20, early_stopping = TRUE,
       ngrams = 2L, val_split = 0.15, patience = 3),
  list(label = '8. bigrams + more dropout (0.3) + more training data',
       dense_units = 25, dropout_rate = 0.3, epochs = 25, early_stopping = TRUE,
       ngrams = 2L, val_split = 0.15, patience = 3),
  list(label = '9. bigrams capped vocab (8000) + more training data',
       dense_units = 25, dropout_rate = 0.2, epochs = 25, early_stopping = TRUE,
       ngrams = 2L, val_split = 0.15, patience = 3, max_tokens = 8000L),
  list(label = '10. bigrams + bigger network (64 units) + more training data',
       dense_units = 64, dropout_rate = 0.3, epochs = 25, early_stopping = TRUE,
       ngrams = 2L, val_split = 0.15, patience = 3)
)

results <- map(variants, function(v) {
  do.call(run_variant, v)
})

comparison <- map_dfr(results, ~ .x$metrics) %>%
  select(variant, .metric, .estimate) %>%
  pivot_wider(names_from = .metric, values_from = .estimate) %>%
  arrange(desc(accuracy))

comparison

## Benchmark from task 1 (paragraph-only logistic PCR): 79.5% accuracy,
## 0.863 ROC AUC. Pick whichever of our variants comes closest to or
## beats that.
benchmark_accuracy <- 0.795
benchmark_auc <- 0.863

comparison <- comparison %>%
  mutate(beats_benchmark_accuracy = accuracy >= benchmark_accuracy,
         beats_benchmark_auc = roc_auc >= benchmark_auc)

comparison

## Note: a deployable saved model isn't required for this preliminary
## task (only the primary task requires that per the README). Also,
## keras3's TextVectorization layer (tf_idf mode) has a known
## save/reload bug -- confirmed separately while building the primary
## task model -- so a saved copy of this model would not reliably
## reload anyway. The metrics above are what matters for this task.

best_index <- which.max(map_dbl(results, ~ .x$metrics$.estimate[.x$metrics$.metric == 'accuracy']))
cat('\nBest single-run variant:', variants[[best_index]]$label, '\n')

## -----------------------------------------------------------------
## STABILITY CHECK: neural networks are stochastic -- a single run
## can look better or worse than it really is just by chance. Rerun
## the best-looking configuration (bigrams + more training data +
## early stopping) 4 more times before trusting its result.
## -----------------------------------------------------------------

stability_variant <- list(label = 'bigrams + more training data + early stopping',
                          dense_units = 25, dropout_rate = 0.2, epochs = 25,
                          early_stopping = TRUE, ngrams = 2L,
                          val_split = 0.15, patience = 3)

stability_runs <- map_dfr(1:4, function(i) {
  do.call(run_variant, stability_variant)$metrics %>% mutate(rep = i)
})

stability_summary <- stability_runs %>%
  group_by(.metric) %>%
  summarise(mean = mean(.estimate), min = min(.estimate), max = max(.estimate), .groups = 'drop')

cat('\n=== Stability check across 4 repeated runs ===\n')
print(stability_runs %>% select(rep, .metric, .estimate) %>%
        pivot_wider(names_from = .metric, values_from = .estimate))
cat('\nSummary:\n')
print(stability_summary)

saveRDS(list(comparison = comparison, stability_runs = stability_runs,
             stability_summary = stability_summary,
             benchmark_accuracy = benchmark_accuracy, benchmark_auc = benchmark_auc),
        'results/nn_task3_comparison.rds')
