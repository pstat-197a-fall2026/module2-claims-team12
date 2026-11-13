# Primary task (a): binary classification -- round 3
#
# Two new things to try, both genuinely different from what we've done:
#
# 1. Regularized logistic regression (glmnet) on the FULL vocabulary
#    (28,146 words), not a truncated top-2000 like our SVM attempt.
#    glmnet is built to handle this many features efficiently via
#    sparse matrices, and automatically shrinks less-useful words
#    toward zero instead of us guessing a cutoff by hand.
#
# 2. Give the model MORE INFORMATION to begin with: include page
#    headers/titles, not just paragraph text. Every experiment so far
#    tweaked how the model learns from the same limited input; this
#    changes what it's allowed to see in the first place.

library(tidyverse)
library(tidymodels)
library(tidytext)
library(textstem)
library(rvest)
library(qdapRegex)
library(glmnet)
library(Matrix)
library(keras3)
library(tensorflow)

class_metrics <- metric_set(sensitivity, specificity, accuracy, roc_auc)

## -----------------------------------------------------------------
## EXPERIMENT 1: glmnet logistic regression, full vocabulary
## -----------------------------------------------------------------

load('data/claims-clean-binary.RData')
set.seed(110122)
partitions <- claims_clean %>% initial_split(prop = 0.8)
train_df <- training(partitions)
test_df  <- testing(partitions)

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
train_idf <- train_tokens %>%
  bind_tf_idf(term = token.lem, document = .id, n = n) %>%
  distinct(token.lem, idf)

vocab <- sort(unique(train_tokens$token.lem))   # full vocabulary, no cap
vocab_index <- setNames(seq_along(vocab), vocab)

# build a sparse (document x word) tf-idf matrix directly, so we never
# materialize a dense 1,573 x 28,146 table in memory
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

  sparseMatrix(
    i = doc_index[tokens$.id],
    j = vocab_index[tokens$token.lem],
    x = tokens$tf_idf,
    dims = c(length(doc_ids), length(vocab_index)),
    dimnames = list(doc_ids, names(vocab_index))
  )
}

x_train <- build_sparse(train_tokens, vocab_index, train_idf)
x_test  <- build_sparse(test_tokens, vocab_index, train_idf)

y_train <- train_df %>% filter(.id %in% rownames(x_train)) %>%
  arrange(match(.id, rownames(x_train))) %>% pull(bclass)
y_test <- test_df %>% filter(.id %in% rownames(x_test)) %>%
  arrange(match(.id, rownames(x_test))) %>% pull(bclass)

# 5-fold cross-validation to choose the regularization strength
set.seed(110122)
cv_glmnet <- cv.glmnet(x_train, y_train, family = 'binomial', alpha = 0, nfolds = 5)
# alpha = 0 (ridge): with 28k words, keep all but shrink most toward
# zero, rather than lasso's harder all-or-nothing word selection

glmnet_probs <- predict(cv_glmnet, newx = x_test, s = 'lambda.min', type = 'response')[, 1]
glmnet_pred <- factor(glmnet_probs > 0.5, labels = levels(y_test))

glmnet_results <- tibble(truth = y_test, pred = glmnet_pred, prob = glmnet_probs) %>%
  class_metrics(truth = truth, estimate = pred, prob, event_level = 'second') %>%
  mutate(variant = 'glmnet ridge logistic regression (full 28k vocab)')

glmnet_results

## -----------------------------------------------------------------
## EXPERIMENT 2: add headers/titles to the scraped text, rerun best NN
## -----------------------------------------------------------------

parse_fn_headers <- function(.html){
  read_html(.html) %>%
    html_elements('p, h1, h2, h3, h4, h5, h6, title') %>%
    html_text2() %>%
    str_c(collapse = ' ') %>%
    rm_url() %>%
    rm_email() %>%
    str_remove_all('\'') %>%
    str_replace_all(paste(c('\n', '[[:punct:]]', 'nbsp', '[[:digit:]]', '[[:symbol:]]'),
                          collapse = '|'), ' ') %>%
    str_replace_all("([a-z])([A-Z])", "\\1 \\2") %>%
    tolower() %>%
    str_replace_all("\\s+", " ")
}

load('data/claims-raw.RData')

claims_clean_headers <- claims_raw %>%
  filter(str_detect(text_tmp, '<!')) %>%
  rowwise() %>%
  mutate(text_clean = parse_fn_headers(text_tmp)) %>%
  unnest(text_clean)

save(claims_clean_headers, file = 'data/claims-clean-headers.RData')

set.seed(110122)
partitions_h <- claims_clean_headers %>% initial_split(prop = 0.8)

train_text_h <- training(partitions_h) %>% pull(text_clean)
train_labels_h <- training(partitions_h) %>% pull(bclass) %>% as.numeric() - 1
test_text_h <- testing(partitions_h) %>% pull(text_clean)
test_labels_h <- testing(partitions_h) %>% pull(bclass) %>% as.numeric() - 1

preprocess_layer_h <- layer_text_vectorization(
  standardize = NULL, split = 'whitespace', ngrams = NULL,
  max_tokens = NULL, output_mode = 'tf_idf'
)
preprocess_layer_h %>% adapt(train_text_h)

model_h <- keras_model_sequential() %>%
  preprocess_layer_h() %>%
  layer_dropout(0.2) %>%
  layer_dense(units = 25) %>%
  layer_dropout(0.2) %>%
  layer_dense(1) %>%
  layer_activation(activation = 'sigmoid')

model_h %>% compile(
  loss = 'binary_crossentropy', optimizer = 'adam', metrics = 'binary_accuracy'
)

model_h %>% fit(
  train_text_h, train_labels_h,
  validation_split = 0.3, epochs = 20,
  callbacks = list(callback_early_stopping(
    monitor = 'val_loss', patience = 2, restore_best_weights = TRUE
  )),
  verbose = 0
)

probs_h <- predict(model_h, test_text_h, verbose = 0) %>% as.numeric()
pred_h <- as.numeric(probs_h > 0.5)

headers_results <- tibble(
  truth = factor(test_labels_h, levels = c(0, 1), labels = c('N/A', 'Relevant')),
  pred  = factor(pred_h, levels = c(0, 1), labels = c('N/A', 'Relevant')),
  prob  = probs_h
) %>%
  class_metrics(truth = truth, estimate = pred, prob, event_level = 'second') %>%
  mutate(variant = 'NN + headers/titles (early stopping)')

headers_results

## -----------------------------------------------------------------
## FULL COMPARISON
## -----------------------------------------------------------------

prior_results <- readRDS('results/binary_advanced_comparison.rds') %>%
  pivot_longer(-variant, names_to = '.metric', values_to = '.estimate')

all_results <- bind_rows(prior_results, glmnet_results, headers_results)

final_comparison <- all_results %>%
  select(variant, .metric, .estimate) %>%
  pivot_wider(names_from = .metric, values_from = .estimate) %>%
  arrange(desc(accuracy))

final_comparison

saveRDS(final_comparison, 'results/binary_round3_comparison.rds')
