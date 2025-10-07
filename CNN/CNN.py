import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

selected_classes = {0: 'airplane', 1: 'car', 7: 'horse', 8: 'ship', 9: 'truck'}

mask_train = np.isin(y_train, list(selected_classes.keys())).flatten()
mask_test = np.isin(y_test, list(selected_classes.keys())).flatten()

x_train, y_train = x_train[mask_train], y_train[mask_train]
x_test, y_test = x_test[mask_test], y_test[mask_test]

# Normalize
x_train, x_test = x_train / 255.0, x_test / 255.0

# Relabel classes to 0–4
label_map = {c: i for i, c in enumerate(selected_classes.keys())}
y_train = np.vectorize(label_map.get)(y_train.flatten())
y_test = np.vectorize(label_map.get)(y_test.flatten())

# CNN model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(5, activation='softmax')  # 5 classes
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=10,
                    validation_data=(x_test, y_test))

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest accuracy: {test_acc:.2f}")

img_paths = [
    "my_images/bus.bmp"
]

custom_images = []
for path in img_paths:
    img = image.load_img(path, target_size=(32, 32))   # resize to 32x32
    img_array = image.img_to_array(img) / 255.0
    custom_images.append(img_array)

custom_images = np.array(custom_images)

predictions = model.predict(custom_images)
class_names = list(selected_classes.values())

for i, path in enumerate(img_paths):
    img = image.load_img(path)
    plt.imshow(img)
    plt.axis("off")
    pred_label = class_names[np.argmax(predictions[i])]
    confidence = np.max(predictions[i]) * 100
    plt.title(f"Predicted: {pred_label} ({confidence:.2f}%)")
    plt.show()

os.makedirs("bitmaps/train", exist_ok=True)
for i in range(10):  # just save first 10 images as example
    img = Image.fromarray((x_train[i]*255).astype(np.uint8))
    img.save(f"bitmaps/train/sample_{i}.bmp")
