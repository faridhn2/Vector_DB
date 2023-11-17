

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

class MyVectorDB():
  summaries = []
  docs = []
  doc_names = []
  sent_model = SentenceTransformer('bert-base-nli-mean-tokens')
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

  def process(self):
    file_names = os.listdir(self.PDFDIR)

    for file_name in file_names:
        reader = PdfReader(os.path.join(self.PDFDIR,file_name))
        number_of_pages = len(reader.pages)
        text = ''
        for i in range(number_of_pages):
            page = reader.pages[i]
            text += page.extract_text()+'\n'

        normal_text = self.preprocess_text(text)

        normal_text = list(set(normal_text))
        sentences = sent_tokenize(' '.join(normal_text))
        self.docs.append(sentences)
        self.doc_names.append(file_name)
        summary = self.generate_summary(text)
        self.summaries.append(summary)
    self.sentence_embeddings = self.sent_model.encode(self.docs)



    self.d = self.sentence_embeddings.shape[1]
    self.index = faiss.IndexFlatL2(self.d)
    self.index.add(self.sentence_embeddings)
  def restart(self):
    self.summaries = []
    self.docs = []
    self.doc_names = []
    self.sentence_embeddings = []
    self.process()
  def search(self,search_text,number_of_result=1):
    k = number_of_result
    xq = self.sent_model.encode([search_text])
    D, I = self.index.search(xq, k)
    return np.array(self.doc_names)[I[0]][0]
  def get_doc_list(self):
    text2 = ''
    for idx , dc in enumerate(self.doc_names):
      text2 += f'{idx} - { dc} \n'
    print(text2)
    return text2
  def get_summary(self,idx):
    text3 = self.doc_names[idx]+' : \n'
    text3 += self.summaries[idx]
    print(text3)
    return text3

# vdb4 = MyVectorDB()
# vdb4.search('Random Forest')

# vdb4.get_doc_list()

# vdb4.get_summary(0)




