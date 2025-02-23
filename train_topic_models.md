# Code to train topic models from raw text files


## NLP with udpipe

First, it is assumed that the files are stored in the `corpus` subdirectory. Then, the package `udpipe` needs to be installed. In the example below, we'll be using the model `portuguese-bosque` to perform all NLP tasks on our collection of Brazilian novels. We activate the library `udpipe`, load the model to a temporary directory, and then we create two empty files: one that will contain resulting text samples (after NLP treatment) and another that will store the information about the file (its name, to be specific):

``` {r}
library(udpipe)
udmodel = udpipe_download_model(language = "portuguese-bosque", 
                                 model_dir = tempdir())

text_files = list.files(path = "corpus")

cat("", file = "texts_lemmatized_N-V-A-A.txt")
cat("", file = "text_IDs_N-V-A-A.txt")

```

Now the most difficult part of the procedure follows. It loops over all the text files from the directory `corpus`, and then for each document, it runs the `udpipe()` function that lemmatizes the texts. Not only this, since it also detects named entities (proper nouns) that will be excluded from the texts a few lines later. Finally, all the lemmatized documents are written (one document in a line) to a joint text file `texts_lemmatized_N-V-A-A.txt`. However, since our texts might be quite long at times -- and many real-life topic models are indeed based on long texts -- one additional step is added to the pipeline, namely each lemmatized text is split into 1000-word chunks. The splitting procedure might seem somewhat obscure, but it basically runs another loop and iteratively writes the next chunk to a joint file `texts_lemmatized_N-V-A-A.txt`. As the name of the file suggests, the pre-processing will also involve selecting only the words that belong to one of the following grammatical categories: Nouns, Verbs, Adjectives, Adverbs:


``` {r}

for(current_file in text_files) {

    current_file = paste("corpus/", current_file, sep = "")

    # let's say something on screen
    message(current_file)

    # lead the next text from file, keep it split into pars
    current_text = readLines(current_file, encoding = "UTF-8", warn = FALSE)
    current_text = paste(current_text, collapse = " ")

    # run udpipe
    parsed_text = udpipe(x = current_text, object = udmodel)

    # get rid of proper nouns and punctuation
    lemmatized_text = parsed_text$lemma[parsed_text$upos == "NOUN" | parsed_text$upos == "VERB" | parsed_text$upos == "ADJ" | parsed_text$upos == "ADV"]


    # get rid of NAs (just in case)
    lemmatized_text = lemmatized_text[!is.na(lemmatized_text)]


    # splitting the string of lemmas into chunks of 1000 words each:
    chunk_size = 1000
    # defining the number of possible chunks
    no_of_chunks = floor(length(lemmatized_text) / chunk_size)

    # writing each chunk into a final txt file
    for(i in 0:(no_of_chunks -1)) {
    
        start_point = (i * chunk_size) + 1
        current_chunk = lemmatized_text[start_point : (start_point + chunk_size)]
        current_chunk = paste(current_chunk, collapse = " ")
        write(current_chunk, file = "texts_lemmatized_N-V-A-A.txt", sep = "\n", append = TRUE)
        chunk_ID = paste(gsub(".txt", "", current_file), sprintf("%03d", i), sep = "_")
        write(chunk_ID, file = "text_IDs_N-V-A-A.txt", sep = "\n", append = TRUE)

    }

}

```

In an alternative scenario, we lemmatized all the words, and excluded all proper nouns (or the ones that belonged to the `PROPN` category). Mind that we used a different output files to store the results:


``` {r}

for(current_file in text_files) {

    current_file = paste("corpus/", current_file, sep = "")

    # let's say something on screen
    message(current_file)

    # lead the next text from file, keep it split into pars
    current_text = readLines(current_file, encoding = "UTF-8", warn = FALSE)
    current_text = paste(current_text, collapse = " ")

    # run udpipe
    parsed_text = udpipe(x = current_text, object = udmodel)

    # get rid of proper nouns and punctuation
    lemmatized_text = parsed_text$lemma[parsed_text$upos != "PROPN"]

    # get rid of NAs (just in case)
    lemmatized_text = lemmatized_text[!is.na(lemmatized_text)]


    # splitting the string of lemmas into chunks of 1000 words each:
    chunk_size = 1000
    # defining the number of possible chunks
    no_of_chunks = floor(length(lemmatized_text) / chunk_size)

    # writing each chunk into a final txt file
    for(i in 0:(no_of_chunks -1)) {
    
        start_point = (i * chunk_size) + 1
        current_chunk = lemmatized_text[start_point : (start_point + chunk_size)]
        current_chunk = paste(current_chunk, collapse = " ")
        write(current_chunk, file = "texts_lemmatized.txt", sep = "\n", append = TRUE)
        chunk_ID = paste(gsub(".txt", "", current_file), sprintf("%03d", i), sep = "_")
        write(chunk_ID, file = "text_IDs.txt", sep = "\n", append = TRUE)

    }

}

```

Finally, in the third scenario we lemmatized the words and took into account only those that were recognized as either Nouns, or Verbs:


``` {r}

for(current_file in text_files) {

    current_file = paste("corpus/", current_file, sep = "")

    # let's say something on screen
    message(current_file)

    # lead the next text from file, keep it split into pars
    current_text = readLines(current_file, encoding = "UTF-8", warn = FALSE)
    current_text = paste(current_text, collapse = " ")

    # run udpipe
    parsed_text = udpipe(x = current_text, object = udmodel)

    # get rid of proper nouns and punctuation
    lemmatized_text = parsed_text$lemma[parsed_text$upos == "NOUN" | parsed_text$upos == "VERB"]


    # get rid of NAs (just in case)
    lemmatized_text = lemmatized_text[!is.na(lemmatized_text)]


    # splitting the string of lemmas into chunks of 1000 words each:
    chunk_size = 1000
    # defining the number of possible chunks
    no_of_chunks = floor(length(lemmatized_text) / chunk_size)

    # writing each chunk into a final txt file
    for(i in 0:(no_of_chunks -1)) {
    
        start_point = (i * chunk_size) + 1
        current_chunk = lemmatized_text[start_point : (start_point + chunk_size)]
        current_chunk = paste(current_chunk, collapse = " ")
        write(current_chunk, file = "texts_lemmatized_Noun-Verb.txt", sep = "\n", append = TRUE)
        chunk_ID = paste(gsub(".txt", "", current_file), sprintf("%03d", i), sep = "_")
        write(chunk_ID, file = "text_IDs_Noun-Verb.txt", sep = "\n", append = TRUE)

    }

}

```

From now on, all the procedures covered in this document should be performed independently for the three scenarios above. The code is identical, the only exception is the input files (e.g. `text_IDs_Noun-Verb.txt`) that need to be updated accordingly. Below, we discuss the scenario with Nouns, Verbs, Adjectives, and Adverbs, which is the scenario covered in the paper.



## Table of word frequencies


First we read the already created files `texts_lemmatized_N-V-A-A.txt` and `text_IDs_N-V-A-A.txt`, followed by combining them into a data frame. For the sake of convenience, these files don't need to be generated from scratch using `udpipe` (see the above code), since the resulting files can be found in the current repository:

``` {r}
raw_data = readLines("texts_lemmatized_N-V-A-A.txt", encoding = "UTF-8")
text_IDs = readLines("text_IDs_N-V-A-A.txt")
data_with_IDs = data.frame(doc_id = text_IDs, text = raw_data)
```

Also, we use a stopword list from the package `tidystopwords`. This stopword will not affect the scenarios with Nouns, Verbs, and other grammatical classes, but let's keep this step for the sake of consistence across the three scenarios discussed in this study:

``` {r}
library(tidystopwords)
stopwords = generate_stoplist(language = "Portuguese")
```

As for now, we're ready to activate the package `tm`, followed by creating a corpus from the existing data frame `text_IDs_N-V-A-A`. The package `tm` will remove punctuation, stopwords, and it will be used to generate the table of frequencies:


``` {r}
library(tm)
corpus = Corpus(DataframeSource(data_with_IDs))

parsed_corpus = tm_map(corpus, content_transformer(tolower))
parsed_corpus = tm_map(parsed_corpus, removeWords, stopwords)
parsed_corpus = tm_map(parsed_corpus, removePunctuation, preserve_intra_word_dashes = TRUE)
parsed_corpus = tm_map(parsed_corpus, removeNumbers)
parsed_corpus = tm_map(parsed_corpus, stripWhitespace)

min_frequency = 5
doc_term_matrix = DocumentTermMatrix(parsed_corpus, 
    control = list(bounds = list(global = c(min_frequency, Inf))))
```

In order to save lots of time, it is wise to save the resulting doc-term matrix (frequency table) into a file:

``` {r}
save(doc_term_matrix, file = "doc_term_matrix_N-V-A-A.RData")
```

All the resulting tables of frequencies can be found in the current repository, in the folder `doc_term_matrices`.



## Optimal number of topics


In order to guess the optimal parameter _k_, which sets the number of topics that we want the model to capture, we might want to iteratively train several independent models in a grid-search mode, where the parameter _k_ is gradually increased, and the quality of the resulting models is evaluated. We use the library `ldatuning` to do the job. Please be warned that executing the following code takes a few dozen hours on a solid machine. It is recommended to delegate this part of the procedure to a high-performence server. 

Before we conduct the actual training, first we remove additional stopwords (as discussed in the paper). Certainly, without this step, the procedure will work as well.


``` {r}
load("doc_term_matrix_N-V-A-A.RData")

more_stopwords = grep("[[:punct:]]", colnames(doc_term_matrix))
doc_term_matrix = doc_term_matrix[, -c(more_stopwords)]

further_stopwords = c("fazer", "dar", "dizer", "perguntar", "replicar", 
    "responder", "falar", "retrucar", "afinal", "agora", "ainda", "ali", "antes",
    "apenas", "assim", "caro", "dentro", "depois", "diante", "dois", "enfim", "entanto",
    "então", "és", "hein", "hoje", "lá", "logo", "primeiro", "quase", "quatro", "sim",
    "só", "somente", "talvez", "também", "tão", "teu", "três", "tua", "último", "um",
    "vosmecê")
unwanted_columns = colnames(doc_term_matrix) %in% further_stopwords
doc_term_matrix = doc_term_matrix[, -c(which(unwanted_columns))]

```

Now, actual training follows. We train 15 models, with _k_ = 10, 30, 50, ..., 300.


``` {R}
library(ldatuning)

results = FindTopicsNumber(
  doc_term_matrix,
  topics = seq(from = 10, to = 300, by = 20),
  metrics = c("Griffiths2004", "CaoJuan2009", "Arun2010", "Deveaud2014"),
  method = "Gibbs",
  control = list(seed = 1234),
  #mc.cores = 2L,
  verbose = TRUE
)
```

It is strongly recommended to save the results, since it took so many hours to compute them:

``` {R}
save(results, file = "TD_k_optimize_for_10-300_topics_N-V-A-A.RData")
```

It is recommended to save the results also for the sake of convenience: having calculated the stuff on a server, we can copy the above file and visualize the results on any local computer, even on a slow laptop:


``` {R}
library(ldatuning)

load("TD_k_optimize_for_10-300_topics_N-V-A-A.RData")
FindTopicsNumber_plot(results)

```



## Training a topic model



Finally, the main procedure follows, with some real-life parameters of the model specified as arguments of the LDA function. Even if a bit redundant, the code begins with the same step of removing extra stopwords, as described above:

``` {r}
library(topicmodels)

load("doc_term_matrix_N-V-A-A.RData")

more_stopwords = grep("[[:punct:]]", colnames(doc_term_matrix))
doc_term_matrix = doc_term_matrix[, -c(more_stopwords)]

further_stopwords = c("fazer", "dar", "dizer", "perguntar", "replicar", 
    "responder", "falar", "retrucar", "afinal", "agora", "ainda", "ali", "antes",
    "apenas", "assim", "caro", "dentro", "depois", "diante", "dois", "enfim", "entanto",
    "então", "és", "hein", "hoje", "lá", "logo", "primeiro", "quase", "quatro", "sim",
    "só", "somente", "talvez", "também", "tão", "teu", "três", "tua", "último", "um",
    "vosmecê")
unwanted_columns = colnames(doc_term_matrix) %in% further_stopwords
doc_term_matrix = doc_term_matrix[, -c(which(unwanted_columns))]
```

The main procedure of training a topic models follows:


``` {R}
number_of_topics = 80
#
topic_model = LDA(doc_term_matrix, k = number_of_topics, method = "Gibbs", 
                  control = list(seed = 1234, burnin = 100, thin = 100, iter = 1000, verbose = 1))
```

The final model can be saved to a file:

``` {R}
save(topic_model, file = "topic_model_k-80_c_N-V-A-A.RData")
```

All the trained topic models used in our study, can be found in the current repository, in the folder `resulting_topic_models`. 


## Wordcloud visualizations


The following code produces a series of wordclouds, in order to visualize the proportions of words in particular topics. First, one needs to create a subfolder, say named `topic_visualizations`. Please then change the working directory, and execute the following code:

``` {r eval = FALSE}
library(wordcloud)

load("../topic_model_k-80_c_N-V-A-A.RData")
# extracting topic distributions
model_weights = posterior(topic_model)
topic_words = model_weights$terms
doc_topics = model_weights$topics

for(topic_id in 1:80) {
  topic_name = paste(names(sort(topic_words[topic_id,], decreasing = TRUE)[1:3]), collapse = "_")
  filename = paste(topic_id, "_", topic_name, ".png", sep = "")
  no_of_words = 50
  current_topic = sort(topic_words[topic_id,], decreasing = TRUE)[1:no_of_words]
  png(file = filename)
  wordcloud(names(current_topic), current_topic, random.order = FALSE, rot.per = 0)
  dev.off()
}
```


## Classification

The classification step, including the SHAP values visualization, will be covered in a separate R script, in the current repository.

