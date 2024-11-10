import math


def split_sentences_to_words(documents):
    return [sentence.split() for sentence in documents]


def remove_commas_and_dots(documents):
    documents_no_commas = [[word.replace(",", "") for word in doc] for doc in documents]
    documents_no_commas_and_dots = [[word.replace(".", "") for word in doc] for doc in documents_no_commas]
    return documents_no_commas_and_dots


def calculate_tf(document):
    tf_dict = {}
    words = document
    length_of_document = len(words)
    for word in words:
        tf_dict[word] = words.count(word) / length_of_document
    return tf_dict


def calculate_idf(documents):
    number_of_documents = len(documents)
    idf_dict = {}
    all_words = set(word for doc in documents for word in doc)

    for word in all_words:
        doc_count = sum(1 for doc in documents if word in doc)
        idf_dict[word] = math.log(
            (number_of_documents / (doc_count + 1)) + 1)
    return idf_dict


def calculate_tfidf(documents):
    documents = split_sentences_to_words(documents)
    documents = remove_commas_and_dots(documents)
    tfidf_docs = []
    idf_dict = calculate_idf(documents)

    for doc in documents:
        tf_dict = calculate_tf(doc)
        tfidf = {word: tf * idf_dict[word] for word, tf in tf_dict.items()}
        tfidf_docs.append(tfidf)

    return list(tfidf_docs)


def get_highest_value_from_tfidf(dict1, dict2):
    common_words = set(dict1.keys()) & set(dict2.keys())
    values = []
    for word in common_words:
        values.append(dict1[word])
        values.append(dict2[word])
    if len(values) == 0:
        return 0
    return max(values)



