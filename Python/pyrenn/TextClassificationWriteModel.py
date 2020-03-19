import tensorflow as tf
from tensorflow import keras
import numpy as np

data = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=10000)

word_index = data.get_word_index()
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# preprocessing data to make it consistent (different lengths for different reviews)
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post",
                                                        maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)

# model
x = [-1, 0, 1, 2, 3, 4]
y = [-3, -1, 1, 3, 5, 7]

model = tf.keras.models.Sequential(
    [tf.keras.layers.Dense(units=1, input_shape=(1000,))])


# model = keras.Sequential()
model.add(keras.layers.Embedding(10000, 16))
# model.add(keras.layers.GlobalAveragePooling1D(data_format='channels_first'))
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))

model.compile(optimizer='sgd', loss='mean_squared_error')
model.fit(x, y, epochs=50)

# model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

x_val = train_data[:10000]
x_train = train_data[:10000]

y_val = train_labels[:10000]
y_train = train_labels[:10000]

# fitModel = model.fit(x_train, y_train, epochs=4, batch_size=512, validation_data=(x_val, y_val), verbose=1)

# results = model.evaluate(test_data, test_labels)


model.save("model.h5")

# Save outputs to certain file
keras_file = "./SavedNN/model_text_classification.hdf5"
keras.models.save_model(model, keras_file)


# convert keras file to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
print(np.shape(model))
print(converter)
tflite_model = converter.convert()
open("./SavedNN/model_text_classification.tflite", "wb").write(tflite_model)


def review_encode(s):
    encoded = [1]

    for word in s:
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2)
    return encoded


