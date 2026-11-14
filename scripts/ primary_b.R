# Primary task (b): multiclass classification
# Run from the project root: Rscript "scripts/ primary_b.R"
library(rvest)
library(quanteda)
library(Matrix)
library(glmnet)

# 1. Load the data and extract titles, headings, and paragraphs
load("data/claims-raw.RData")

clean_text <- function(html) {
  vapply(html, function(page) {
    document <- read_html(charToRaw(enc2utf8(page)), encoding = "UTF-8")
    text <- paste(html_text2(html_elements(document, "title, h1, h2, h3, h4, h5, h6, p")),
                  collapse = " ")
    text <- gsub("https?://[^[:space:]]+|[^[:space:]]+@[^[:space:]]+", " ", text)
    tolower(text)
  }, character(1), USE.NAMES = FALSE)
}

word_counts <- function(text) {
  words <- tokens(text, remove_punct = TRUE, remove_numbers = TRUE,
                  remove_symbols = TRUE)
  dfm(tokens_ngrams(words, n = 1:2))
}

counts <- word_counts(clean_text(claims_raw$text_tmp))
labels <- claims_raw$mclass

# 2. Split subjects into training (60%), tuning (20%), and validation (20%)
# Pages from the same subject stay together.
set.seed(197212)
subjects <- sample(unique(claims_raw$neo_search_subject_id))
n <- length(subjects)
train_subjects <- subjects[1:floor(0.6 * n)]
tune_subjects <- subjects[(floor(0.6 * n) + 1):floor(0.8 * n)]
train <- which(claims_raw$neo_search_subject_id %in% train_subjects)
tune <- which(claims_raw$neo_search_subject_id %in% tune_subjects)
validation <- which(!claims_raw$neo_search_subject_id %in% c(train_subjects, tune_subjects))

# 3. Convert counts to TF-IDF using the fitted vocabulary and IDF weights
tfidf <- function(counts, idf) {
  x <- as(dfm_match(counts, features = names(idf)), "dgCMatrix")
  x@x <- 1 + log(x@x)
  x <- x %*% Diagonal(x = as.numeric(idf))
  lengths <- sqrt(rowSums(x * x))
  lengths[lengths == 0] <- 1  # Empty pages stay zero vectors.
  Diagonal(x = 1 / lengths) %*% x
}

# Learn text features and fit ridge multinomial regression on the given rows.
lambdas <- 10^seq(-2, -5, length.out = 7)
fit_model <- function(rows) {
  training_counts <- dfm_trim(counts[rows, ], min_docfreq = 3)
  terms <- names(sort(docfreq(training_counts), decreasing = TRUE))
  training_counts <- dfm_match(training_counts, features = head(terms, 20000))
  idf <- log((1 + nrow(training_counts)) / (1 + docfreq(training_counts))) + 1
  fit <- glmnet(tfidf(training_counts, idf), labels[rows],
                family = "multinomial", alpha = 0, lambda = lambdas,
                standardize = FALSE)
  list(fit = fit, idf = idf)
}

# 4. Select the penalty using tuning accuracy
model <- fit_model(train)
tuning_predictions <- predict(model$fit, newx = tfidf(counts[tune, ], model$idf),
                               s = lambdas, type = "class")
tuning_accuracy <- colMeans(tuning_predictions == as.character(labels[tune]))
lambda <- lambdas[which.max(tuning_accuracy)]
print(data.frame(Lambda = lambdas, Accuracy = tuning_accuracy))

# 5. Refit on training + tuning pages and evaluate on validation pages
model <- fit_model(c(train, tune))
predicted <- as.character(predict(model$fit,
  newx = tfidf(counts[validation, ], model$idf), s = lambda, type = "class"))
confusion <- table(Actual = labels[validation],
                   Predicted = factor(predicted, levels = levels(labels)))
tp <- diag(confusion)
fn <- rowSums(confusion) - tp
fp <- colSums(confusion) - tp
tn <- sum(confusion) - tp - fn - fp
per_class <- data.frame(Class = levels(labels),
                        Sensitivity = tp / (tp + fn), Specificity = tn / (tn + fp))
metrics <- data.frame(Accuracy = sum(tp) / sum(confusion),
                      Sensitivity = mean(per_class$Sensitivity),
                      Specificity = mean(per_class$Specificity))
majority <- names(which.max(table(labels[c(train, tune)])))
baseline_accuracy <- mean(labels[validation] == majority)
print(confusion)
print(metrics)
print(per_class)

# 6. Train the final model on all labeled pages and save it
model <- fit_model(seq_len(nrow(claims_raw)))
model$lambda <- lambda
model$classes <- levels(labels)
model$metrics <- metrics
model$per_class <- per_class
model$baseline_accuracy <- baseline_accuracy
model$split_sizes <- c(Training = length(train), Tuning = length(tune),
                       Validation = length(validation))
saveRDS(model, "results/multiclass-model.rds")

# 7. Deliverable 4: use the saved model to generate test predictions
# For predictions without retraining, run scripts/multiclass-predict.R.
load("data/claims-test.RData")
model <- readRDS("results/multiclass-model.rds")
test_counts <- word_counts(clean_text(claims_test$text_tmp))
predicted <- as.character(predict(model$fit, newx = tfidf(test_counts, model$idf),
                                 s = model$lambda, type = "class"))
pred_df <- data.frame(.id = claims_test$.id,
                      mclass.pred = factor(predicted, levels = model$classes))
# Deliverable 3: multiclass portion; merge with Part (a) by .id for preds-group12.RData.
save(pred_df, file = "results/preds-multiclass-group12.RData")
