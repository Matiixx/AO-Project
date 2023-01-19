import tensorflow as tf


def trainModel():
  train_dataset = tf.keras.utils.image_dataset_from_directory(
      'LicensePlatesCharacters',
      seed=2137,
      labels="inferred",
      label_mode="categorical",
      image_size=(128, 64),
      batch_size=64,
      subset='training',
      validation_split=0.2,
      shuffle=True,
      color_mode="rgb",
  )

  val_dataset = tf.keras.utils.image_dataset_from_directory(
      'LicensePlatesCharacters',
      seed=420,
      labels="inferred",
      label_mode="categorical",
      image_size=(128, 64),
      batch_size=64,
      subset='validation',
      validation_split=0.2,
      shuffle=True,
      color_mode="rgb",
  )

  class_names = train_dataset.class_names
  model = tf.keras.models.Sequential()

  model.add(tf.keras.layers.RandomFlip("horizontal_and_vertical"))
  model.add(tf.keras.layers.Conv2D(
    24, 5, activation='relu', input_shape=(128, 64, 3)))
  model.add(tf.keras.layers.MaxPooling2D())
  model.add(tf.keras.layers.Conv2D(48, 5, activation='relu'))
  model.add(tf.keras.layers.MaxPooling2D())
  model.add(tf.keras.layers.Conv2D(96, 5, activation='relu', ))
  model.add(tf.keras.layers.MaxPooling2D())
  model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dropout(0.4))
  model.add(tf.keras.layers.Dense(len(class_names), activation='softmax',))

  model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
  model.fit(
      train_dataset,
      validation_data=val_dataset,
      workers=10,
      epochs=10,
      batch_size=64
  )
  model.save('licensePlatesReader.model')


trainModel()
