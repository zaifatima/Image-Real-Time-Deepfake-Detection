import os
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# Set parameters
image_size = 128
batch_size = 32
epochs = 10
dataset_path = 'Dataset'
checkpoint_dir = 'checkpoints'
architecture_path = 'cnn_architecture.json'

# Ensure checkpoint directory exists
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, 'cnn_epoch_{epoch:02d}.weights.h5')  # ✅ fixed extension

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255)
val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(dataset_path, 'train'),
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = val_test_datagen.flow_from_directory(
    os.path.join(dataset_path, 'validation'),
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='categorical'
)
print(train_generator.class_indices)

# Load existing model or build new
latest_checkpoint = None
if os.path.exists(architecture_path):
    print("🔄 Loading model architecture...")
    with open(architecture_path, 'r') as f:
        model = model_from_json(f.read())

    # Load last checkpoint if available
    checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.weights.h5')])
    if checkpoints:
        latest_checkpoint = os.path.join(checkpoint_dir, checkpoints[-1])
        print(f"✅ Found checkpoint: {latest_checkpoint}")
        model.load_weights(latest_checkpoint)
    else:
        print("⚠️ No checkpoints found. Starting fresh.")
else:
    print("🆕 Building new model...")
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(image_size, image_size, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dense(2, activation='softmax')
    ])
    with open(architecture_path, 'w') as f:
        f.write(model.to_json())

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Callback to save after every epoch
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    verbose=1
)

# Train
print("🚀 Training model...")
model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=epochs,
    callbacks=[checkpoint_callback]
)

# ✅ Save final weights
model.save_weights("cnn_final.weights.h5")  # Final snapshot

# Optional: Save full model (structure + weights)
# model.save("cnn_final_full_model.h5")
