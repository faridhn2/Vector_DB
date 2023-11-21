

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
  summaries = []
  docs = []
  doc_names = []
  doc_text =[]
  vectorizer = TfidfVectorizer()
  PDFDIR = 'pdfs'
  def __init__(self):
    self.process()
  def preprocess_text(self,text):
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


    words1 = self.preprocess_text(sent1)
    words2 = self.preprocess_text(sent2)

    all_words = list(set(words1 + words2))

    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    for word in words1:
        vector1[all_words.index(word)] += 1

    for word in words2:
        vector2[all_words.index(word)] += 1

    return 1 - cosine_distance(vector1, vector2)

  def generate_summary(self,text, num_sentences=2):
    sentences = sent_tokenize(text)

    sentence_scores = []

    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i == j:
                continue
            score = self.sentence_similarity(sentences[i], sentences[j])
            sentence_scores.append((i, j, score))

    sentence_scores.sort(key=lambda x: x[2], reverse=True)

    summary_sentences = set()
    summary_sentences.add(0)  # Include the first sentence

    for i in range(len(sentence_scores)):
        summary_sentences.add(sentence_scores[i][1])
        if len(summary_sentences) == num_sentences:
            break


    summary = " ".join(sentences[i] for i in summary_sentences)
    return summary
  def generate_summary2(self,text, per):
    nlp = spacy.load('en_core_web_sm')
    doc= nlp(text)
    tokens=[token.text for token in doc]
    word_frequencies={}
    for word in doc:
        if word.text.lower() not in list(STOP_WORDS):
            if word.text.lower() not in punctuation:
                if word.text not in word_frequencies.keys():
                    word_frequencies[word.text] = 1
                else:
                    word_frequencies[word.text] += 1
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
    select_length=int(len(sentence_tokens)*per)
    summary=nlargest(select_length, sentence_scores,key=sentence_scores.get)
    final_summary=[word.text for word in summary]
    summary=''.join(final_summary)
    return summary
  def process(self):
    if not os.path.exists(self.PDFDIR):
      os.makedirs(self.PDFDIR)
    file_names = os.listdir(self.PDFDIR)
    before_id = 0
    
    for file_name in file_names:
      if '.pdf' in file_name :
        
        reader = PdfReader(os.path.join(self.PDFDIR,file_name))
        number_of_pages = len(reader.pages)
        text = ''
        text_for_sum = ''
        for i in range(number_of_pages):
            page = reader.pages[i]
            text += page.extract_text()+'\n'
            text = text.lower()
            text_for_sum += text
            # normal_text = self.preprocess_text(text)

            # normal_text = list(set(normal_text))
            sentences = sent_tokenize(text)
            self.docs.extend(sentences)
        self.doc_names.append((file_name,range(before_id,len(self.docs))))
        before_id = len(self.docs)
        self.doc_text.append(text_for_sum)
        # summary = self.generate_summary(text_for_sum)
        # self.summaries.append(summary)
    self.sentence_embeddings = self.vectorizer.fit_transform(self.docs)



    # self.d = self.sentence_embeddings.shape[1]
    # self.index = faiss.IndexFlatL2(self.d)
    # self.index.add(self.sentence_embeddings)
  def restart(self):
    self.summaries = []
    self.docs = []
    self.doc_names = []
    self.sentence_embeddings = []
    self.doc_text = []
    self.process()
  def search(self,search_text,number_of_result=5):
    search_text = search_text.lower()
    k = number_of_result
    xq = self.vectorizer.transform([search_text]).toarray()
    results = cosine_similarity(self.sentence_embeddings,xq)
    results = pd.DataFrame(results)
    res_d = {}
    for s_id in list(results[0].nlargest(n=k).index):
      for r in self.doc_names:
        if s_id in r[1]:
          if r[0] in res_d:
            res_d[r[0]]+=1
          else:
            res_d[r[0]]=1
          break
    res_d = sorted(res_d.items(),key=lambda x:x[1],reverse=True)
    print(res_d)
    star = '*'
    return '\n'.join(list(map(lambda x:f'{star*x[1]} {x[0]}',res_d)))
  def get_doc_list(self):
    text = ''
    for idx , dc in enumerate(self.doc_names):
      text += f'{idx} - { dc[0]} \n'
    print(text)
    return text
  def get_summary(self,idx):

    text = self.doc_names[idx][0]+' : \n'
    # text += self.summaries[idx]
    # print(text)
    sum_text = self.doc_text[idx]
    summary = self.generate_summary(sum_text)
    text += summary
    return text

# vdb4 = MyVectorDB()
# vdb4.search('Random Forest')

# vdb4.get_doc_list()

# vdb4.get_summary(0)





