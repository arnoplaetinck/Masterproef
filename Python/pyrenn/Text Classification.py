import tensorflow as tf
from tensorflow import keras
import tflite
import numpy as np

data = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=10000)

print(train_data[0])

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

model = keras.Sequential()
model.add(keras.layers.Embedding(10000, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))


model.compile(optimizer="adam", loss="binary_crossentropy",metrics=["accuracy"])

x_val = train_data[:10000]
x_train = train_data[:10000]

y_val = train_labels[:10000]
y_train = train_labels[:10000]

fitModel = model.fit(x_train, y_train, epochs=4, batch_size=512, validation_data=(x_val, y_val), verbose=1)

results = model.evaluate(test_data, test_labels)


model.save("model.h5")

# Save outputs to certain file
keras_file = "./SavedNN/model_text_classification.h5"
keras.models.save_model(model, keras_file)


# convert keras file to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
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



'''
model = keras.models.load_model("model.h5")

with open("test.txt", encoding="utf-8") as f:
    for line in f.readlines():
        nline = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").replace("\"",
                                                                                                                  "").strip(
            " ")
        encode = review_encode(nline)
        encode = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post",
                                                            maxlen=250)
        predict = model.predict(encode)
        print(line)
        print(encode)
        print(predict[0])
'''