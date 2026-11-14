# Deliverable 4, Part (b): predict from the saved model without retraining.
# Run from the project root: Rscript scripts/multiclass-predict.R
library(rvest)
library(quanteda)
library(Matrix)
library(glmnet)

model <- readRDS("results/multiclass-model.rds")
load("data/claims-test.RData")

# Use the same text extraction and word pairs as the training script.
text <- vapply(claims_test$text_tmp, function(page) {
  document <- read_html(charToRaw(enc2utf8(page)), encoding = "UTF-8")
  text <- paste(html_text2(html_elements(document, "title, h1, h2, h3, h4, h5, h6, p")),
                collapse = " ")
  text <- gsub("https?://[^[:space:]]+|[^[:space:]]+@[^[:space:]]+", " ", text)
  tolower(text)
}, character(1), USE.NAMES = FALSE)
words <- tokens(text, remove_punct = TRUE, remove_numbers = TRUE,
                remove_symbols = TRUE)
counts <- dfm(tokens_ngrams(words, n = 1:2))

# Apply the saved vocabulary and IDF weights, then normalize each page.
x <- as(dfm_match(counts, features = names(model$idf)), "dgCMatrix")
x@x <- 1 + log(x@x)
x <- x %*% Diagonal(x = as.numeric(model$idf))
lengths <- sqrt(rowSums(x * x))
lengths[lengths == 0] <- 1
x <- Diagonal(x = 1 / lengths) %*% x

predicted <- as.character(predict(model$fit, newx = x,
                                 s = model$lambda, type = "class"))
pred_df <- data.frame(.id = claims_test$.id,
                      mclass.pred = factor(predicted, levels = model$classes))
save(pred_df, file = "results/preds-multiclass-group12.RData")
