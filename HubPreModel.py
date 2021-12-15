import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras import layers
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import tensorflow_hub as hub

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

# We can use this encoding layer in place of our text_vectorizer and embedding layer
sentence_encoder_layer = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4",
                                        input_shape=[], # shape of inputs coming to our model
                                        dtype=tf.string, # data type of inputs coming to the USE layer
                                        trainable=False, # keep the pretrained weights (we'll create a feature extractor)
                                        name="USE")

# Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_sparse_categorical_accuracy', min_lr=0.001, patience=5, mode='min', verbose=1)
early_stopping = EarlyStopping(patience=5, monitor='val_sparse_categorical_accuracy')
callbacks = [early_stopping]

# Create model using the Sequential API
model_hub = tf.keras.Sequential([sentence_encoder_layer, # take in sentences and then encode them into an embedding
                                 layers.Dense(64, activation="relu"),
                                 layers.Dense(5, activation='softmax')
], name="model_hub")

# Compile model
model_hub.compile(optimizer=tf.keras.optimizers.Adam(),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

# Fit the model
history_hub = model_hub.fit(train_sentences,
                              train_labels,
                              epochs=5,
                              callbacks=callbacks,
                              validation_data=(val_sentences, val_labels))

np.save('Auswertung/history_hub.npy', history_hub.history)

import matplotlib.pyplot as plt
pd.DataFrame(history_hub.history).plot()
plt.ylabel("loss")
plt.xlabel("epochs")
plt.show()

model_hub.save('Auswertung/model_hub')