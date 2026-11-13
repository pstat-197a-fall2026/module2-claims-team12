# Deliverable 4 (Part A, binary classification): use the saved model
# to generate predictions on new data, without retraining.
#
# Run from the project root: Rscript scripts/varun/binary-predict.R
#
# NOTE: the saved model (results/binary-final-model.keras) contains
# only the trainable dense layers, not the text-vectorization step --
# keras3's TextVectorization layer (tf_idf mode) cannot currently be
# reliably saved/reloaded (confirmed bug: save_model()/load_model()
# silently fail to restore its learned vocabulary). The vectorization
# step is deterministic given the same training text, so it is rebuilt
# here (fast, a few seconds) before loading the saved model.

library(tidyverse)
library(keras3)
library(tensorflow)

source('scripts/preprocessing.R')

## Rebuild the same text-vectorization step used during training
load('data/claims-raw.RData')
claims_clean <- claims_raw %>% parse_data()
all_text <- claims_clean %>% pull(text_clean)

preprocess_layer <- layer_text_vectorization(
  standardize = NULL, split = 'whitespace', ngrams = NULL,
  max_tokens = NULL, output_mode = 'tf_idf'
)
preprocess_layer %>% adapt(all_text)

## Load the saved model
model <- load_model('results/binary-final-model.keras')

## Load new data to predict on (here: the provided test set)
load('data/claims-test.RData')

safe_parse <- function(html) {
  tryCatch(parse_fn(html), error = function(e) '')
}

clean_new <- claims_test %>%
  rowwise() %>%
  mutate(text_clean = safe_parse(text_tmp)) %>%
  ungroup() %>%
  select(.id, text_clean)

x_new <- preprocess_layer(clean_new$text_clean)
probs <- predict(model, as.matrix(x_new), verbose = 0) %>% as.numeric()

bclass_levels <- levels(claims_raw$bclass)
pred_df <- tibble(
  .id = clean_new$.id,
  bclass.pred = factor(probs > 0.5, labels = bclass_levels)
)

print(table(pred_df$bclass.pred))
