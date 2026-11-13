# Primary task (a): binary classification -- baseline model
#
# Goal: get a first, honest accuracy number using the example pipeline
# from scripts/nlp-model-development.R, ported to keras3 (the modern
# Keras interface) since the old keras package wasn't installed and
# keras3 is the currently maintained one. Same architecture and training
# setup as the example script; this is our starting point to improve on.

library(tidyverse)
library(tidymodels)
library(keras3)
library(tensorflow)

source('scripts/preprocessing.R')

## PREPROCESSING
#################

load('data/claims-raw.RData')

claims_clean <- claims_raw %>%
  parse_data()

save(claims_clean, file = 'data/claims-clean-binary.RData')

## PARTITION
#############

set.seed(110122)
partitions <- claims_clean %>%
  initial_split(prop = 0.8)

train_text <- training(partitions) %>% pull(text_clean)
train_labels <- training(partitions) %>% pull(bclass) %>% as.numeric() - 1

test_text <- testing(partitions) %>% pull(text_clean)
test_labels <- testing(partitions) %>% pull(bclass) %>% as.numeric() - 1

## MODEL: TF-IDF features -> small neural network
###################################################

preprocess_layer <- layer_text_vectorization(
  standardize = NULL,
  split = 'whitespace',
  ngrams = NULL,
  max_tokens = NULL,
  output_mode = 'tf_idf'
)

preprocess_layer %>% adapt(train_text)

model <- keras_model_sequential() %>%
  preprocess_layer() %>%
  layer_dropout(0.2) %>%
  layer_dense(units = 25) %>%
  layer_dropout(0.2) %>%
  layer_dense(1) %>%
  layer_activation(activation = 'sigmoid')

summary(model)

model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = 'binary_accuracy'
)

history <- model %>%
  fit(train_text,
      train_labels,
      validation_split = 0.3,
      epochs = 5)

## EVALUATE ON HELD-OUT TEST SET
##################################

test_probs <- predict(model, test_text) %>% as.numeric()
test_pred <- as.numeric(test_probs > 0.5)

results_df <- tibble(
  truth = factor(test_labels, levels = c(0, 1), labels = c('N/A', 'Relevant')),
  pred  = factor(test_pred,   levels = c(0, 1), labels = c('N/A', 'Relevant')),
  prob  = test_probs
)

class_metrics <- metric_set(sensitivity, specificity, accuracy, roc_auc)

baseline_metrics <- results_df %>%
  class_metrics(truth = truth, estimate = pred, prob, event_level = 'second') %>%
  mutate(variant = 'baseline NN (keras3 port)')

baseline_metrics

## SAVE MODEL + RESULTS
#########################

save_model(model, 'results/binary-baseline-model.keras')
saveRDS(baseline_metrics, 'results/binary_baseline_metrics.rds')
