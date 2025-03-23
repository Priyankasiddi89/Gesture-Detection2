import torch
import open_clip
import cv2
import numpy as np
from PIL import Image

# Check if GPU is available
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device="cpu"
print(f"Using device: {device}")

# Load CLIP model and move it to GPU
model, preprocess, tokenizer = open_clip.create_model_and_transforms(
    'ViT-L-14-quickgelu', pretrained='openai'
)
model.to(device)  # Move model to GPU

# Define action labels and detailed descriptions
action_descriptions = {
    "punching": ["a person is violently punching another person"],
    "kicking": ["a person is aggressively kicking another person"],
    "slapping": ["a person is slapping another person aggressively"],
    "pushing": ["a person is forcefully pushing another person"],
    "shouting": ["a person is yelling aggressively with wide open mouth"],
    "falling": ["a person is falling down"],
    "running": ["a person is running away in panic"],
    "normal": ["a person is behaving normally"]
}

# Flatten descriptions and store corresponding labels
descriptions = []
labels = []

for action, desc_list in action_descriptions.items():
    descriptions.extend(desc_list)
    labels.extend([action] * len(desc_list))  # Ensure matching index

# Tokenize descriptions and move to GPU
text_tokens = open_clip.tokenize(descriptions).to(device)

# Open webcam
cap = cv2.VideoCapture(0)  # Change to (1) if the first camera doesn't work

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Process webcam frames continuously
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame.")
        break

    # Convert frame to PIL Image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image_tensor = preprocess(image).unsqueeze(0).to(device)  # Move image to GPU

    # Get CLIP embeddings
    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        text_features = model.encode_text(text_tokens)

    # Compute similarity scores
    similarity = (image_features @ text_features.T).softmax(dim=-1)

    # Get the best description match
    best_match_index = similarity.argmax().item()
    best_action = labels[best_match_index]  # Get corresponding action label
    best_score = similarity.max().item()

    # Confidence threshold
    threshold = 0.7
    if best_score < threshold:
        best_action = "normal"

    # Overlay results on the frame
    cv2.putText(frame, f"{best_action} ({best_score:.2f})", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the frame with detections
    cv2.imshow("Action Recognition", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

print("Real-time action recognition completed.")
