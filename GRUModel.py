import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras import layers
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

train_df = pd.read_table("Data/train.tsv")
train_df = train_df.drop(["PhraseId", "SentenceId"], axis=1)
print(train_df.head())

train_df_shuffled = train_df.sample(frac=1, random_state=42) # shuffle with random_state=42 for reproducibility
print(train_df_shuffled.head())

# Use train_test_split to split training data into training and validation sets
train_sentences, val_sentences, train_labels, val_labels = train_test_split(train_df_shuffled["Phrase"].to_numpy(),
                                                                            train_df_shuffled["Sentiment"].to_numpy(),
                                                                            test_size=0.1,
                                                                            random_state=42)

# Check
#print(len(train_sentences))
#print(len(train_labels))
#print(len(val_sentences))
#print(len(val_labels))
#print(train_sentences[:10])
#print(train_labels[:10])

# Find average number of tokens (words) in training Tweets
#print(round(sum([len(i.split()) for i in train_sentences])/len(train_sentences)))

# Setup text vectorization variables
max_vocab_length = 10000  # max number of words to have in our vocabulary
max_length = 8  # max length our sequences will be (e.g. how many words from a Tweet does our model see?)

text_vectorizer = TextVectorization(max_tokens=max_vocab_length,
                                    output_mode="int",
                                    output_sequence_length=max_length)

embedding = layers.Embedding(input_dim=max_vocab_length,  # set input shape
                             output_dim=512,  # set size of embedding vector
                             embeddings_initializer="uniform",  # default, intialize randomly
                             input_length=max_length)  # how long is each input

# Adapt vectorizer
text_vectorizer.adapt(train_sentences)

# Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_sparse_categorical_accuracy', min_lr=0.001, patience=5, mode='min', verbose=1)
early_stopping = EarlyStopping(patience=5, monitor='val_sparse_categorical_accuracy')
callbacks = [early_stopping,reduce_lr]

# Build an RNN using the GRU cell
from tensorflow.keras import layers
inputs = layers.Input(shape=(1,), dtype="string")
x = text_vectorizer(inputs)
x = embedding(x)
x = layers.GRU(64, return_sequences=True)(x) # stacking recurrent cells requires return_sequences=True
x = layers.GRU(64)(x)
x = layers.Dense(64, activation="relu")(x) # optional dense layer after GRU cell
outputs = layers.Dense(5, activation=None)(x)
model_gru = tf.keras.Model(inputs, outputs, name="model_GRU")

# Compile model
model_gru.compile(optimizer=tf.keras.optimizers.Adam(),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

# Fit the model
history_gru = model_gru.fit(train_sentences,
                              train_labels,
                              epochs=5,
                              callbacks=callbacks,
                              validation_data=(val_sentences, val_labels))

np.save('Auswertung/history_gru.npy', history_gru.history)

import matplotlib.pyplot as plt
pd.DataFrame(history_gru.history).plot()
plt.ylabel("loss")
plt.xlabel("epochs")
plt.show()

model_gru.save('Auswertung/model_gru')