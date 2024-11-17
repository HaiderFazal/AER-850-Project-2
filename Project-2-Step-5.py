from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Step 5: Model Testing

model = load_model("haider_model.h5")

model.summary()

def preprocess_image(image_path, target_size=(500, 500)):

    img = load_img(image_path, target_size=target_size)

    img_array = img_to_array(img)
   
    img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis=0)
    return img_array, img

test_images = {
    "crack": {
        "path": "C:/Users/Saira/Desktop/Uni/Year 4/Sem 1/AER-850/Project-2/Project 2 Data/Data/test/crack/test_crack.jpg", 
        "label": "crack",
    },
    "missing_head": {
        "path": "C:/Users/Saira/Desktop/Uni/Year 4/Sem 1/AER-850/Project-2/Project 2 Data/Data/test/missing-head/test_missinghead.jpg",
        "label": "missing-head",
    },
    "paint_off": {
        "path": "C:/Users/Saira/Desktop/Uni/Year 4/Sem 1/AER-850/Project-2/Project 2 Data/Data/test/paint-off/test_paintoff.jpg",
        "label": "paint-off",
    },
}

class_names = ["crack", "missing-head", "paint-off"]

fig, axes = plt.subplots(1, len(test_images), figsize=(15, 5))

for i, (defect_type, details) in enumerate(test_images.items()):

    processed_image, original_image = preprocess_image(details["path"])

    predictions = model.predict(processed_image)

    predicted_class_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_index]

    probabilities = [
        f"{class_names[j]}: {predictions[0][j]*100:.1f}%"
        for j in range(len(class_names))
    ]

    axes[i].imshow(original_image)
    axes[i].axis("off")
    axes[i].set_title(
        f"True Classification : {details['label']}\nPredicted Classification: {predicted_class}",
        fontsize=10,
        color="black",
    )

    text = "\n".join(probabilities)
    axes[i].text(
        0.5,
        -0.15,
        text,
        fontsize=9,
        color="green",
        backgroundcolor="white",
        transform=axes[i].transAxes,
        ha="center",
    )

plt.tight_layout()
plt.show()

