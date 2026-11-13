# Primary task (a): FINAL binary classification model
#
# Winning approach from experimentation (see scripts/varun/binary-*.R):
# TF-IDF + small neural network, trained with early stopping to avoid
# overfitting. Held-out test accuracy during comparison: 0.808
# (sensitivity 0.823, specificity 0.791, ROC AUC 0.842), the best and
# most robust result across every variant tried (tuning, dropout
# combos, SVM, glmnet, header-augmented text, threshold tuning).
#
# This script retrains that approach on ALL labeled data (not just the
# 80% split used for comparison), so the deployed model learns from
# every available labeled page before being used on the real unlabeled
# test set.
#
# NOTE on saving: keras3 (this TF/Keras version) has a known bug
# serializing a TextVectorization layer with output_mode = 'tf_idf' --
# save_model()/load_model() fail to restore it (confirmed directly).
# Fix: the text-vectorization step is kept OUTSIDE the saved model.
# We save only the trainable dense layers; scripts/varun/binary-predict.R
# rebuilds the (deterministic) vectorization step from the training
# text before loading the saved model to generate predictions.

library(tidyverse)
library(tidymodels)
library(keras3)
library(tensorflow)

source('scripts/preprocessing.R')

load('data/claims-raw.RData')
claims_clean <- claims_raw %>% parse_data()

## Small validation slice so early stopping still has something to
## monitor, even though we're otherwise using (almost) all labeled data.
set.seed(110122)
final_split <- claims_clean %>% initial_split(prop = 0.9)
train_final <- training(final_split)
val_final   <- testing(final_split)

train_text <- train_final %>% pull(text_clean)
train_labels <- train_final %>% pull(bclass) %>% as.numeric() - 1
val_text <- val_final %>% pull(text_clean)
val_labels <- val_final %>% pull(bclass) %>% as.numeric() - 1

## Preprocessing layer: fit (adapt) on ALL labeled text, so the
## deployed vocabulary is as complete as possible.
all_text <- claims_clean %>% pull(text_clean)
preprocess_layer <- layer_text_vectorization(
  standardize = NULL, split = 'whitespace', ngrams = NULL,
  max_tokens = NULL, output_mode = 'tf_idf'
)
preprocess_layer %>% adapt(all_text)

x_train <- preprocess_layer(train_text)
x_val   <- preprocess_layer(val_text)

final_model <- keras_model_sequential(input_shape = ncol(x_train)) %>%
  layer_dropout(0.2) %>%
  layer_dense(units = 25) %>%
  layer_dropout(0.2) %>%
  layer_dense(1) %>%
  layer_activation(activation = 'sigmoid')

final_model %>% compile(
  loss = 'binary_crossentropy', optimizer = 'adam', metrics = 'binary_accuracy'
)

final_model %>% fit(
  as.matrix(x_train), train_labels,
  validation_data = list(as.matrix(x_val), val_labels),
  epochs = 20,
  callbacks = list(callback_early_stopping(
    monitor = 'val_loss', patience = 2, restore_best_weights = TRUE
  )),
  verbose = 0
)

## Save the deployable model (dense layers only -- see note above)
save_model(final_model, 'results/binary-final-model.keras', overwrite = TRUE)

## Generate predictions on the real, unlabeled test set
##
## NOTE: parse_data()'s filter(str_detect(text_tmp, '<!')) silently
## drops 14 of 929 test rows -- checked individually: 7 are genuine
## scraping failures (e.g. literal "Error: ..." text, not HTML), 4
## parse fine but have no <p> content, and 3 are valid pages that
## simply lack a doctype declaration and WOULD have usable text. Since
## every one of the 929 test pages needs a prediction for the final
## deliverable, we parse every row directly (skipping that filter) and
## fall back to an empty string (-> model's default/majority-leaning
## prediction) only for rows that truly fail to parse.
load('data/claims-test.RData')

safe_parse <- function(html) {
  tryCatch(parse_fn(html), error = function(e) '')
}

clean_test <- claims_test %>%
  rowwise() %>%
  mutate(text_clean = safe_parse(text_tmp)) %>%
  ungroup() %>%
  select(.id, text_clean)

stopifnot(nrow(clean_test) == nrow(claims_test))   # guarantee all 929 covered

x_test <- preprocess_layer(clean_test$text_clean)
test_probs <- predict(final_model, as.matrix(x_test), verbose = 0) %>% as.numeric()
bclass_levels <- levels(claims_raw$bclass)
bclass_pred <- factor(test_probs > 0.5, labels = bclass_levels)

bclass_pred_df <- tibble(.id = clean_test$.id, bclass.pred = bclass_pred)

save(bclass_pred_df, file = 'results/preds-binary-group12.RData')

cat('Final binary model trained and predictions saved.\n')
cat('Predicted class distribution on test set (n =', nrow(bclass_pred_df), '):\n')
print(table(bclass_pred_df$bclass.pred))
