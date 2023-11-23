

from nltk.tokenize import sent_tokenize

from PyPDF2 import PdfReader
import faiss
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.cluster.util import cosine_distance
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from sentence_transformers import SentenceTransformer
import os
import numpy as np
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
class MyVectorDB():
  """
    A class for handling a vectorized document database.

    Attributes:
    - summaries: List to store generated summaries.
    - docs: List to store processed documents.
    - doc_names: List to store document names and corresponding document ranges.
    - doc_text: List to store the text content of each document.
    - vectorizer: TfidfVectorizer for vectorizing text data.
    - PDFDIR: Directory to store PDF files.

    Methods:
    - __init__: Initializes the class and processes documents.
    - preprocess_text: Tokenizes, removes stopwords, stems, and lemmatizes the input text.
    - sentence_similarity: Calculates cosine similarity between two sentences.
    - generate_summary: Generates a summary using sentence similarity.
    - generate_summary2: Generates a summary using word frequency and spacy.
    - process: Processes PDF documents in the specified directory.
    - restart: Resets the class attributes.
    - search: Searches for documents related to the input text.
    - get_doc_list: Returns a formatted list of document names.
    - get_summary: Returns a document summary based on its index.
  """
  
  def __init__(self):
    """
        Initializes the MyVectorDB class.

        This method sets up the class attributes and processes documents.

        Parameters:
        - None

        Returns:
        - None
    """
    
    self.summaries = []
    self.docs = []
    self.doc_names = []
    self.doc_text =[]
    self.vectorizer = TfidfVectorizer()
    self.PDFDIR = 'pdfs'
    # main process
    self.process()

  def preprocess_text(self,text):

    """
        Preprocesses the input text.

        This method performs tokenization, stopword removal, stemming, and lemmatization.

        Parameters:
        - text (str): The input text to be preprocessed.

        Returns:
        - list: A list of preprocessed tokens.
    """

    # Tokenization
    tokens = word_tokenize(text)

    # Stopword Removal
    stop_words = set(stopwords.words('english'))

    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in stemmed_tokens]

    return lemmatized_tokens

  def sentence_similarity(self,sent1, sent2):
    """
        Calculates the cosine similarity between two sentences.

        Parameters:
        - sent1 (str): The first sentence.
        - sent2 (str): The second sentence.

        Returns:
        - float: Cosine similarity score between the two sentences.
    """

    # preprocess sentences
    words1 = self.preprocess_text(sent1)
    words2 = self.preprocess_text(sent2)

    all_words = list(set(words1 + words2))
    # voctorize sentences
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    for word in words1:
        vector1[all_words.index(word)] += 1

    for word in words2:
        vector2[all_words.index(word)] += 1
    # return cosine similarity score of these two sentences' vectors
    return 1 - cosine_distance(vector1, vector2)

  def generate_summary(self,text, num_sentences=2):
    """
        Generates a summary using sentence similarity.

        Parameters:
        - text (str): The input text for summarization.
        - num_sentences (int): Number of sentences in the summary (default is 2).

        Returns:
        - str: The generated summary.
    """
    # tokenize sentences
    sentences = sent_tokenize(text)
    # compute cosine similarity score of all sentences
    sentence_scores = []

    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i == j:
                continue
            score = self.sentence_similarity(sentences[i], sentences[j])
            sentence_scores.append((i, j, score))

    # sort for finding best scores
    sentence_scores.sort(key=lambda x: x[2], reverse=True)

    summary_sentences = set()
    summary_sentences.add(0)  # Include the first sentence
    # find best n sentences based on scores
    for i in range(len(sentence_scores)):
        summary_sentences.add(sentence_scores[i][1])
        if len(summary_sentences) == num_sentences:
            break

    summary = " ".join(sentences[i] for i in summary_sentences)
    return summary

  def generate_summary2(self,text, per):
    """
        Generates a summary using word frequency and spacy.

        Parameters:
        - text (str): The input text for summarization.
        - per (float): Percentage of sentences to include in the summary.

        Returns:
        - str: The generated summary.
    """
    # load summarizing model
    nlp = spacy.load('en_core_web_sm')

    doc= nlp(text)
    tokens=[token.text for token in doc]
    # find word frequencies
    word_frequencies={}
    for word in doc:
        if word.text.lower() not in list(STOP_WORDS):
            if word.text.lower() not in punctuation:
                if word.text not in word_frequencies.keys():
                    word_frequencies[word.text] = 1
                else:
                    word_frequencies[word.text] += 1
    # find max frequency
    max_frequency=max(word_frequencies.values())

    for word in word_frequencies.keys():
        word_frequencies[word]=word_frequencies[word]/max_frequency
    sentence_tokens= [sent for sent in doc.sents]
    sentence_scores = {}

    for sent in sentence_tokens:
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if sent not in sentence_scores.keys():                            
                    sentence_scores[sent]=word_frequencies[word.text.lower()]
                else:
                    sentence_scores[sent]+=word_frequencies[word.text.lower()]
    # compute length of summary 
    select_length=int(len(sentence_tokens)*per)
    # generate summary
    summary=nlargest(select_length, sentence_scores,key=sentence_scores.get)
    # postprocessing summary
    final_summary=[word.text for word in summary]
    summary=''.join(final_summary)
    return summary

  def process(self):
    """
        Processes PDF documents in the specified directory.

        This method reads PDF files, extracts text, 
        and preprocesses the text for further analysis.

        Parameters:
        - None

        Returns:
        - None
    """
    # find all pdf files in doc directory
    if not os.path.exists(self.PDFDIR):
      os.makedirs(self.PDFDIR)
    file_names = os.listdir(self.PDFDIR)
    before_id = 0
    
    for file_name in file_names:
      if '.pdf' in file_name :
        # create agent for pdf reading
        reader = PdfReader(os.path.join(self.PDFDIR,file_name))
        number_of_pages = len(reader.pages)
        text = ''
        text_for_sum = ''
        # extract text from pdf per each page
        for i in range(number_of_pages):
            page = reader.pages[i]
            text += page.extract_text()+'\n'
            text = text.lower()
            text_for_sum += text
            # normal_text = self.preprocess_text(text)

            # normal_text = list(set(normal_text))
            sentences = sent_tokenize(text)
            self.docs.extend(sentences)
        # save sentences index range for each document
        self.doc_names.append((file_name,range(before_id,len(self.docs))))
        before_id = len(self.docs)
        self.doc_text.append(text_for_sum)
        # summary = self.generate_summary(text_for_sum)
        # self.summaries.append(summary)

    # In this method all of tokenizing, preprocessing and vectorizing has been done 
    # this model return sparse matrix of (n_samples, n_features) 
    
             
    self.sentence_embeddings = self.vectorizer.fit_transform(self.docs)


  def restart(self):
    """
        Resets the class attributes.

        This method clears the existing data and reprocesses documents.

        Parameters:
        - None

        Returns:
        - None
    """
    # reset all parameters and run again process for createing vectore db
    self.summaries = []
    self.docs = []
    self.doc_names = []
    self.sentence_embeddings = []
    self.doc_text = []
    self.process()

  def search(self,search_text,number_of_result=5):
    """
        Searches for documents related to the input text.

        Parameters:
        - search_text (str): The text to search for in the documents.
        - number_of_result (int): Number of search results to return (default is 5).

        Returns:
        - str: Formatted search results.
    """
    # make input lowercase
    search_text = search_text.lower()
    k = number_of_result
    # vectorize search input with same transfomation model
    xq = self.vectorizer.transform([search_text]).toarray()
    # find similarity scores between input vectore and db sentences vectores
    results = cosine_similarity(self.sentence_embeddings,xq)
    # use dataframe for searching faster
    results = pd.DataFrame(results)
    res_d = {}
    # find best result based on frequency of document titles 
    for s_id in list(results[0].nlargest(n=k).index):
      for r in self.doc_names:
        if s_id in r[1]:
          if r[0] in res_d:
            res_d[r[0]]+=1
          else:
            res_d[r[0]]=1
          break
    
    res_d = sorted(res_d.items(),key=lambda x:x[1],reverse=True)
    # print(res_d)
    star = '*'
    return '\n'.join(list(map(lambda x:f'{star*x[1]} {x[0]}',res_d)))

  def get_doc_list(self):
    """
        Returns a formatted list of document names.

        Parameters:
        - None

        Returns:
        - str: Formatted document list.
    """
    text = ''
    for idx , dc in enumerate(self.doc_names):
      text += f'{idx} - { dc[0]} \n'
    # print(text)
    return text
  def get_summary(self,idx):
    """
        Returns a document summary based on its index.

        Parameters:
        - idx (int): Index of the document.

        Returns:
        - str: Document summary.
    """
    text = self.doc_names[idx][0]+' : \n'
   
    sum_text = self.doc_text[idx]
    summary = self.generate_summary(sum_text)
    text += summary
    return text


# Example usage:
# db = MyVectorDB()
# summary = db.get_summary(0)
# print(summary)



