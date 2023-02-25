import numpy as np
import pickle
import streamlit as st
import pandas as pd
# loading the saved model
loaded_model = pickle.load(open('C:/Users/NAVEEN/trained_model.sav', 'rb'))

from googletrans import Translator
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer





def rating_prediction(a):

    stop_words = set(stopwords.words('english'))

    text = a.split()
    ct = " ".join(ch for ch in text if ch.isalnum())
    token_text = word_tokenize(ct.lower())

    fs = [ch for ch in token_text if not ch.lower() in stop_words]
    #print(fs)
    #print(len(fs)) #Filtered Sentence
    fs1 = set(fs)
    fs2 = list(fs1)

    sid = SentimentIntensityAnalyzer()
    pos_list=[]
    neu_list=[]
    neg_list=[]

    for word in fs2:
        if (sid.polarity_scores(word)['compound']) >= 0.5:
            pos_list.append(word)
        elif (sid.polarity_scores(word)['compound']) <= -0.5:
            neg_list.append(word)
        else:
            neu_list.append(word)                

    len_pos = len(pos_list)
    len_neu = len(neu_list)
    len_neg = len(neg_list)

    li = [[len_pos, len_neu, len_neg]]

    lm = pd.DataFrame(li)

    b = loaded_model.predict(lm)
    return(b)

def main():
    st.title("Predict Rating on the basis of Customer Perception")
    z = st.text_input('Kindly enter your perception about hotel')
    #code for Prediction
    c = ''
    #creating a button for Prediction
    if st.button('Show Prediction'):
        c = rating_prediction(z)
        st.success(c[0])
    
   
   
if __name__ == '__main__':
    main()
       