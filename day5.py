from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding ,LSTM,Dense
import matplotlib.pyplot as plt


import numpy as np

#Load the Dataset

vocab_size = 10000
max_len = 200

(X_train,y_train),(X_test,y_test) = imdb.load_data(num_words=vocab_size)

#Decode reviews to text processing
word_index = imdb.get_word_index() # here we will get a Key:Value pair ,ex -> apple:1 


reverse_word_index = {value: key for key,value in word_index.items()} #here we are reversing the key value pair of word_index ,ex -> 1: apple

decode_reviews = ["".join([reverse_word_index.get(i-3,"?") for i in review]) for review in X_train[:5]]
#most important line : here we are taking each review and decoding it (Already encoded dataset) 
#for offest purpose we are substracting each review by 3 and matchs it with the reverse_word_index and 
#If matches substitute it with the word else ?



#Pad Sequences
X_train = pad_sequences(X_train,maxlen = max_len,padding= "post")
X_test = pad_sequences(X_test,maxlen = max_len,padding= "post")

#Load GloVe Embeddings

embedding_index = {}
glove_file = 'glove.6B.100d.txt'


with open (glove_file,"r",encoding='utf-8') as file:
    for line in file:
        values = line.split()
        word = values[0]
        coef = np.asarray(values[1:],dtype='float')
        embedding_index[word] = coef

print(f"Length of Embedding Index : {len(embedding_index)}  word Vectors")


#Prepare embedding matrix
embedding_dim = 100
embedding_matrix = np.zeros((vocab_size,embedding_dim))
for word,i in word_index.items():
    if i < vocab_size:
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i]= embedding_vector


#Defining LSTM Model with GloVe Embedding

model = Sequential([
    Embedding(input_dim = vocab_size,output_dim = embedding_dim,weights=[embedding_matrix],trainable= False),
    LSTM(128,activation='tanh',return_sequences=False),
    Dense(1,activation='sigmoid')
])
#Compile Model

model.compile(optimizer = 'adam',loss= 'binary_crossentropy',metrics=['accuracy'])

model.summary()

history = model.fit(
    X_train,y_train,validation_split=0.2,epochs=5,batch_size=64,verbose=1
)

loss,accuracy = model.evaluate(X_test,y_test,verbose=0)
print(f"LSTM Model with GloVe Test Accuracy: {accuracy:.4f}")


#LSTM model Without GloVe Embedding

lstm_model = Sequential([
    Embedding(input_dim = vocab_size,output_dim = 128),
    LSTM(128,activation = 'tanh',return_sequences =False),
    Dense(1,activation='sigmoid')

])

lstm_model.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics=['accuracy'])
lstm_model.summary()

lstm_history = lstm_model.fit(X_train,y_train,epochs =5,batch_size=64,validation_split=0.2)
lstm_loss,lstm_accuracy = lstm_model.evaluate(X_test,y_test)

print(f"LSTM Test Accuracy : {lstm_accuracy:.4f}")

#Plot Accuracy Comparison

models = ['LSTM','LSTM GloVe']
accuracies = [lstm_accuracy,accuracy]
plt.bar(models,accuracies,color=['blue','green'])
plt.title("Comparison of LSTM model and LSTM with GloVe Embeddings")
plt.ylabel('Accuracy')
plt.show()