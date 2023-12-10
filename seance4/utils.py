
import nltk
import numpy as np
import pandas as pd
import sklearn
import warnings
import re
import string

from pymongo  import MongoClient
from collections import Counter

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from imblearn.under_sampling import RandomUnderSampler

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn import (linear_model, 
                     metrics,
                     preprocessing, 
                     model_selection, 
                     pipeline,
                    neural_network,)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from xgboost import XGBClassifier

from nltk.stem.snowball import FrenchStemmer

###############################Preprocess########################################

def delete_digit(doc):
    return re.sub('[0-9]+', '', doc)

def delete_ponctuation(doc):
    punc = string.punctuation 
    punc += '\n\r\t'
    return doc.translate(str.maketrans(punc, ' ' * len(punc)))

def stem(doc):
  docs_stem = []
  stemmer = FrenchStemmer()
  
  tokens = str(doc).split()

  stemmed_tokens = [stemmer.stem(token) for token in tokens]
  stemmed_text = " ".join(stemmed_tokens)
  
  return stemmed_text
  

def lemmatize(text):
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    lemmatized_text = " ".join(lemmatized_tokens)
    return lemmatized_text


def load_data(preprocessor=True):
    
    client = MongoClient(host="localhost", port=27017)
    db = client["PLDAC"]
    avis = db["avis"]
    df_avis = pd.DataFrame(list(avis.find()))
    #df_avis.dropna(subset=['comment'], inplace=True)
    comments = df_avis['comment'].astype(str)
    notes = df_avis['note'].round()
    
    if preprocessor:
        comments = comments.str.lower()
        comments = comments.map(delete_digit)
        comments = comments.map(delete_ponctuation)
    return comments, notes
    
def words_frequencies(comments):
  
  vectorizer = CountVectorizer()
  bag_of_words = vectorizer.fit_transform(comments)  # creer le bow
  sum_words = bag_of_words.sum(axis=0) # nb occurrences de chaque mot
  words_freq = [(str(word), sum_words[0, idx]) for word, idx in     vectorizer.vocabulary_.items()] # couple (mot, freq)
  words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)  # trie par freq decroissante
  
  return words_freq
   
def binarisation(classes,nb=True):
  if nb:
    n_cls = [1 if note>5 else -1 for note in classes]
  else:
    n_cls = [1 if note<=5 else (2 if note<=7 else 3) for note in classes]

  return n_cls


def top_words(model, vectorizer):
    # Obtenir les caractéristiques les plus importantes pour le meilleur modèle
    if model is None: # Que pour lr et svm
      print('Poids des mots positifs et négatifs non connus, entre guillemets !')
      return
    
    n = 10
    feature_weights = model.coef_[0]
    top_features = {
        'positive': [vectorizer.get_feature_names_out()[i] for i in feature_weights.argsort()[-n:][::-1]],
        'negative': [vectorizer.get_feature_names_out()[i] for i in feature_weights.argsort()[:n]]
    }
    print("Top 10 mots positifs: ", top_features['positive'])
    print("Top 10 mots négatifs: ", top_features['negative'])
    


def classifieur(vectorizer,comments,classes):

    X = vectorizer.fit_transform(comments)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, classes, test_size=0.2, random_state=0)
    rus = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = rus.fit_resample(X_train, y_train)


    # Naïve Bayes
    nb_clf = MultinomialNB()
    nb_clf.fit(X_resampled, y_resampled)

    # Logistic Regression
    lr_clf = LogisticRegression(random_state=0, solver='lbfgs', max_iter=1000, n_jobs=-1)
    lr_clf.fit(X_resampled, y_resampled)

    # Linear SVM
    svm_clf = LinearSVC(random_state=0, tol=1e-5)
    svm_clf.fit(X_resampled, y_resampled)
    
    # Random Forest
    rf_clf = RandomForestClassifier(max_depth=2, random_state=0)
    rf_clf.fit(X_resampled, y_resampled)

    # prediction
    pred_nb = nb_clf.predict(X_test)
    pred_lr = lr_clf.predict(X_test)
    pred_svm = svm_clf.predict(X_test)
    pred_rf = rf_clf.predict(X_test)

    # calcul du F1-score
    nb_f1 = metrics.f1_score(y_test, pred_nb, average='weighted')
    lr_f1 = metrics.f1_score(y_test, pred_lr, average='weighted')
    svm_f1 = metrics.f1_score(y_test, pred_svm, average='weighted')
    rf_f1 = metrics.f1_score(y_test, pred_rf, average='weighted')
    
    
    # Trouver le meilleur modèle en fonction du f1-score
    noms_models = ['Naïve Bayes','Logistic Regression','SVM', 'Random Forest']
    models = [nb_clf,lr_clf,svm_clf,rf_clf]
    f1_scores = [nb_f1,lr_f1,svm_f1,rf_f1] 
    arg_best_model = np.argmax(f1_scores)
    best_model = models[arg_best_model]
    nom_best_model = noms_models[arg_best_model]
    
    
    print(f"Best model: {nom_best_model}")
    print(f"{nom_best_model.capitalize()} accuracy: {metrics.accuracy_score(y_test, best_model.predict(X_test))}  f1-score: {f1_scores[arg_best_model]}")

    if best_model not in (lr_clf,svm_clf):
      best_model = None
    return best_model, vectorizer

    
    
    
def count_vectorizer(comments, classes,nbins=True,**count_vectorizer_args):
  new_classes = binarisation(classes,nb=nbins)
  vectorizer = CountVectorizer(**count_vectorizer_args)
  
  return classifieur(vectorizer, comments, new_classes) 


def tfidf_vectorizer(comments, classes,nbins=True,**tfidf_vectorizer_args):
  new_classes = binarisation(classes,nb=nbins)
  vectorizer = TfidfVectorizer(**tfidf_vectorizer_args)
  
  return classifieur(vectorizer, comments, new_classes)
    
    
    
    
    
    
    
    
    
    
    
    
    
    