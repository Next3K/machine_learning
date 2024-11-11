import math
import numpy as np
from ast import literal_eval


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

    return tfidf_docs


def get_tokens(doc_1, doc_2):
    return list(set(doc_1.keys()).union(set(doc_2.keys())))


def dict_to_vector(doc, tokens):
    vector = np.zeros(len(tokens))
    token_to_idx = {token: idx for idx, token in enumerate(tokens)}
    for token, value in doc.items():
        if token in token_to_idx:
            vector[token_to_idx[token]] = value
    return vector


def cos_sim(vector_1, vector_2):
    dot_product = np.dot(vector_1, vector_2)
    vec_1_len = np.linalg.norm(vector_1)
    vec_2_len = np.linalg.norm(vector_2)
    similarity = dot_product / (vec_1_len * vec_2_len)
    return similarity


def calculate_value_between_docs(doc_1, doc_2):
    doc_1 = literal_eval(doc_1)
    doc_2 = literal_eval(doc_2)
    tokens = get_tokens(doc_1, doc_2)
    vector_1 = dict_to_vector(doc_1, tokens)
    vector_2 = dict_to_vector(doc_2, tokens)
    return cos_sim(vector_1, vector_2)
