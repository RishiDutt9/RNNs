from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
import tensorflow as tf
from tensorflow.keras.layers import Embedding,SimpleRNN,Dense

#Defining the Size and length
vocab_size = 10000
max_len = 200

#Loading the Dataset

(X_train,y_train),(X_test,y_test) = imdb.load_data(num_words = vocab_size) 

#X will be top 10000 frequent words, y is -1 for negative Sentiment,+1 for Positive
# print(f"Size of training Data {X_train.shape}")
# print(f"Size of training Data {X_test.shape}")



X_train = pad_sequences(X_train,maxlen = max_len,padding= "post") #padding = post means padding will happen at the end of the X_train

X_test = pad_sequences(X_test,maxlen = max_len,padding= "post")

# print(f"Size of training Data {X_train.shape}")
# print(f"Size of training Data {X_test.shape}")

model = Sequential([ #Sequential Model stacks layers Sequentially
    Embedding(input_dim = vocab_size,output_dim = 128), #each word index is map to 128 dim vector
    SimpleRNN(128, activation = 'tanh',return_sequences = False), #number of rnn units are 128 ,Output should be last time results
    Dense(1,activation = 'sigmoid')#Fully connected output layers ,1 single output for binary classification 
])

model.compile(
    optimizer = 'adam',
    loss = 'binary_crossentropy',
    metrics =['accuracy']
)

model.summary()

history = model.fit(X_train,y_train,epochs = 5,batch_size = 32,validation_split= 0.2)


loss, accuracy = model.evaluate(X_test,y_test)

print(f"Test loss : {loss:.4f}, Test Accuracy : {accuracy:.4f}")