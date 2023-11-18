An experimental project to extract text from PDF files and create a search mechanism based on tokenization, vectorization, and pre-processing to remove unimportant words and similarity measurement based on cosine distance.
Also, a database vector was created through which the search operation based on the sentences in the PDF files can be easily discovered and the most suitable documents are introduced.
Using two APIs based on GRPC, a client-server mechanism was created to manage commands and transfer new PDF files.
The final file used is also marked with run.py .
The required resources for CI/CD were also created and you can easily use Docker by using the Docker resources available in the project. Docker was tested on Ubuntu 22.0 Linux server.
The following code will help you to use it on coleb.

[colab file](https://colab.research.google.com/drive/1KutHq1yGnoFFoaoXuBOTB4EnoR1ifcIP?usp=sharing)

### commands:
* list : List of docs with indexes
* search search sentences : find best docs for search sentences
* summary doc index : summary of target doc
* restart : after upload new docs we must restart vector database
* file file_path : Upload file to pdfs folder
