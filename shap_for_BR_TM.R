
library(shapviz)
library(ggplot2)
library(xgboost)
library(topicmodels)


set.seed(1234)


load("topic_model_k-80_c_N-V-A-A.RData")
ids = readLines("text_IDs_N-V-A-A.txt")

# extracting topic distributions
model_weights = posterior(topic_model)
topic_words = model_weights$terms
doc_topics = model_weights$topics
rownames(doc_topics) = gsub("corpus/", "", ids)


no_of_topics = dim(topic_words)[1]
topic_names = c()
for(topic_id in 1:no_of_topics) {
  topic_3words = paste(names(sort(topic_words[topic_id,], decreasing = TRUE)[1:3]), collapse = "_")
  topic_name = paste(topic_id, topic_3words, sep = "_")
  topic_names = c(topic_names, topic_name)
}
colnames(doc_topics) = topic_names



# pick a dataset
X = doc_topics

# define the task: either U vs R, or 19 vs 20
#
# Regionalism vs. Urbanism
class_labels = gsub("[0-9].*", "", rownames(doc_topics))
# 19th vs 20th century
#class_labels = gsub(".([0-9]{2})_.*", "\\1", rownames(doc_topics))
#

# replace "U" and "R" (or "19" and "20") with 1 and 0, resp.
class_labels = as.numeric(factor(class_labels)) -1
# reverse the binary values:
#class_labels = as.numeric(factor(class_labels)) *-1 +2



# split the data into the train and the test set
randomize_samples = sample(length(rownames(X)))
class_labels = class_labels[randomize_samples]
X = X[randomize_samples,]
# split into train and test: 80/20
train_size = round(dim(X)[1] * .8)
full_size = dim(X)[1]

X_train = X[1:train_size, ]
labels_train = class_labels[1:train_size]
X_test = X[(train_size +1): full_size, ]
labels_test = class_labels[(train_size +1): full_size]

# turn into xgb objects
dtrain = xgb.DMatrix(data.matrix(X_train), label = labels_train)
dtest = xgb.DMatrix(data.matrix(X_test), label = labels_test)
watchlist = list(eval = dtest, train = dtrain)


# learning rate and other parameters:
# https://www.hackerearth.com/practice/machine-learning/machine-learning-algorithms/beginners-tutorial-on-xgboost-parameter-tuning-r/tutorial/
# https://machinelearningmastery.com/tune-learning-rate-for-gradient-boosting-with-xgboost-in-python/
# Typically, it lies between 0.01 - 0.3

# Fit (untuned) model
fit = xgb.train(
  params = list(learning_rate = 0.1, objective = "binary:logistic"), 
  data = dtrain,
  nrounds = 100,
  eval.metric = "rmse", eval.metric = "logloss", 
  watchlist = watchlist
)


# perform cross-validation
#cv <- xgb.cv(data = dtrain, nrounds = 100, nthread = 2, nfold = 5, metrics = list("rmse","auc"),
#             max_depth = 3, eta = 1, objective = "binary:logistic")
#
#cv$evaluation_log[100,8]
#plot(unlist(cv$evaluation_log[,2]))




# SHAP analysis: X can even contain factors
#X_explain = X[sample(nrow(X), 700), ]
X_explain = X
shp = shapviz(fit, X_pred = data.matrix(X_explain), X = X_explain)



# plot the shap values on screen
#
sv_importance(shp, kind = "bee")
#sv_importance(shp, kind = "bee", viridis_args = list(begin = 0.25, end = 0.85, option = "viridis"))


# produce final plots as PNG files
#
#png(filename = "20-vs-19_results_for_80_topics.png", width = 7, height = 5, units = "in", res = 300)
png(filename = "U-vs-R_results_for_80_topics.png", width = 7, height = 5, units = "in", res = 300)
sv_importance(shp, kind = "bee", viridis_args = list(begin = 0.25, end = 0.85, option = "viridis"))
dev.off()







#### PCA

load("topic_model_k-80_c_N-V-A-A.RData")
ids = readLines("text_IDs_N-V-A-A.txt")


# extracting topic distributions
model_weights = posterior(topic_model)
topic_words = model_weights$terms
doc_topics = model_weights$topics
rownames(doc_topics) = gsub("corpus/", "", ids)



png(filename = "PCA.png", width = 10, height = 5, units = "in", res = 300)

op = par(mfrow = c(1,2))

# Regionalism vs. Urbanism
class_labels = gsub("[0-9].*", "", rownames(doc_topics))
# replace "U" and "R" with explicit color names
class_labels = gsub("R", "green", class_labels)
class_labels = gsub("U", "red", class_labels)

pca_results = prcomp(doc_topics)
expl_var = round(((pca_results$sdev^2) / sum(pca_results$sdev^2) * 100), 1)
PC1_lab = paste("PC1 (", expl_var[1], "%)", sep="")
PC2_lab = paste("PC2 (", expl_var[2], "%)", sep="")

plot(pca_results$x[,1], pca_results$x[,2], 
     col = class_labels,
     xlab = PC1_lab, ylab = PC2_lab,
    )
legend("topright", bty = "n", legend = c("Urbanism", "Regionalism"), col = c("red", "green"), lwd = 5)


# 19th vs 20th century
class_labels = gsub(".([0-9]{2})_.*", "\\1", rownames(doc_topics))
# replace "19" and "20" with explicit color names
class_labels = gsub("19", "blue", class_labels)
class_labels = gsub("20", "brown", class_labels)

pca_results = prcomp(doc_topics)
expl_var = round(((pca_results$sdev^2) / sum(pca_results$sdev^2) * 100), 1)
PC1_lab = paste("PC1 (", expl_var[1], "%)", sep="")
PC2_lab = paste("PC2 (", expl_var[2], "%)", sep="")

plot(pca_results$x[,1], pca_results$x[,2], 
     col = class_labels,
     xlab = PC1_lab, ylab = PC2_lab
    )
legend("topright", bty = "n", legend = c("19th century", "20th century"), col = c("blue", "brown"), lwd = 5)

dev.off()


