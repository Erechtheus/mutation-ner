# Script to transform corpus from .ann to .iob format

import pandas as pd
import urllib.request  # handles url
import spacy # spacing and tokenization #conda install -c conda-forge spacy #python -m spacy download en_core_web_sm

nlp = spacy.load('en_core_web_sm')

url_corpus = "https://raw.githubusercontent.com/Erechtheus/mutationCorpora/master/corpora/original/SETH/corpus.txt"
corpus = urllib.request.urlopen(url_corpus)

corpus_tokenized = []                     # store all tokenized abstracts (list of a list (of abstracts) of a list (of sentences of words))
                                          # [ [ ["w1", "w2"], ["w1", "w2"] ], [ ["w1", "w2"], ["w1", "w2"] ], [ ["w1", "w2"], ["w1", "w2"] ] ]
corpus_iob = []
#corpus_sentenceNumber = []
#sentence_counter = 1

for line in corpus:                       # line = one abstract
      decoded_line = line.decode("utf-8") # ID:0-7 SPACE firstWord
      pubMed_id = decoded_line.split()[0] # save pubMed iD
      print(pubMed_id, '\n')
      abstract = decoded_line[len(pubMed_id)+1:] # remove pubMed ID, so that only abstract text remains
#      print(abstract, '\n')
      doc = nlp(abstract)                 # feed text to language object nlp -> tokenization
      abstract_tokenized = []             # store tokenized abstract divivded in sentences [ ["w1", "w2"], ["w1", "w2"] ], [ ["w1", "w2"], ["w1", "w2"] ]
      abstract_iob = []
#      abstract_sentenceNumber = []
      abstract_tokenized.append('#'+ pubMed_id)
      abstract_iob.append('#'+ pubMed_id)
#      abstract_sentenceNumber.append('#'+ pubMed_id)

      url_annotation = "https://raw.githubusercontent.com/Erechtheus/mutationCorpora/master/corpora/original/SETH/annotations/" + pubMed_id + ".ann"
      annotation = urllib.request.urlopen(url_annotation)
      annotation_label_list = []
      offset_start_list = []
      offset_end_list = []
      for line_annotation in annotation:
            decoded_line_annotation = line_annotation.decode("utf-8")
#            print(decoded_line_annotation, '\n')
            if decoded_line_annotation.split()[0][0] == 'T':
                  annotation_label_list.append(decoded_line_annotation.split()[1])
                  offset_start_list.append(int(decoded_line_annotation.split()[2]))
                  offset_end_list.append(int(decoded_line_annotation.split()[3]))
      
      if len(annotation_label_list) == 0:
            annotation_label_list.append(float('inf'))
            offset_start_list.append(float('inf'))
            offset_end_list.append(float('inf'))

#      print(annotation_label_list, '\n')
#      print(offset_start_list, '\n')
#      print(offset_end_list, '\n')

      i = 0                               # index in abstract
      annotation_i = 0                    # line in annotation file 
      for sent in doc.sents:              # access sentences
            sentence_tokenized = []       # ["w1", "w2"]
            sentence_iob = []
#            sentence_sentenceNumber = []
            intermediate = False

            for token in sent:            # access words/symbols (tokens)
                  if (i >= offset_end_list[annotation_i]) & (annotation_i < len(offset_end_list)-1) :
                        annotation_i += 1
                        intermediate = False
            
                  if (i < offset_start_list[annotation_i]) | (i >= offset_end_list[annotation_i]):  # non-entity case
                        intermediate = False
                        sentence_tokenized.append(token)
                        sentence_iob.append("O")
#                        sentence_sentenceNumber.append("Sentence: " + str(sentence_counter))
                        i += len(token.text_with_ws) # if there is a whitespace after token, count the whitespace as a character too
                       
                  else:                   # entity case
                        if intermediate:  # intermediate case
                              sentence_tokenized.append(token)
                              sentence_iob.append("I-" + annotation_label_list[annotation_i])
#                              sentence_sentenceNumber.append("Sentence: " + str(sentence_counter))
                              i += len(token.text_with_ws)  # if there is a whitespace after token, count the whitespace as a character too
                              
                        else:             # Begin case
                              sentence_tokenized.append(token)
                              sentence_iob.append("B-" + annotation_label_list[annotation_i])
#                              sentence_sentenceNumber.append("Sentence: " + str(sentence_counter))
                              i += len(token.text_with_ws)  # if there is a whitespace after token, count the whitespace as a character too
                              if i < offset_end_list[annotation_i]:
                                    intermediate = True 

            sentence_tokenized.append(" ") # add space after each sentence
            sentence_iob.append(" ")
            # sentence_sentenceNumber.append(" ")
#            sentence_counter += 1
            abstract_tokenized.append(sentence_tokenized) 
            abstract_iob.append(sentence_iob)
#            abstract_sentenceNumber.append(sentence_sentenceNumber)

      corpus_tokenized.append(abstract_tokenized)    
      corpus_iob.append(abstract_iob)   
#      corpus_sentenceNumber.append(abstract_sentenceNumber)       

# print('corpus tokenized: ', '\n')
# print(corpus_tokenized, '\n')
# print('corpus iob: ', '\n')
# print(corpus_iob, '\n')


#dict = {'Sentence #': corpus_sentenceNumber, 'Word': corpus_tokenized, 'Tag': corpus_iob}
dict = {'Word': corpus_tokenized, 'Tag': corpus_iob}
df = pd.DataFrame(dict)

df_ = df.apply(pd.Series.explode).reset_index()
df__ = df_.apply(pd.Series.explode).reset_index()
# df__ = df__.rename(columns={"level_0": "lvl_0"})
# df___ = df__.apply(pd.Series.explode).reset_index()
#output = df___[['Sentence #', 'Word', 'Tag']]
output = df__[['Word', 'Tag']]

#output.to_csv("corpus_IOB_SENT.csv", index=False)
output.to_csv("corpus_IOB.csv", index=False)