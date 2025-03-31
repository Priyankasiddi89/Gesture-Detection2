#final3.1
import torch
import open_clip
import cv2
import numpy as np
from PIL import Image

# Load CLIP model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, preprocess, tokenizer = open_clip.create_model_and_transforms(
    'ViT-L-14-quickgelu', pretrained='openai'
)
model = model.to(device)

# Define Primary Classes
primary_classes = {
    "normal": ["normal"],
    "running": ["running", "sprinting", "jogging"],
    "violence": ["violence"]
}

# Define Secondary Classes (Only if Violence is Detected)
secondary_classes = {
    "punching": ["punching", "throwing a punch"],
    "hitting": ["hitting", "slapping", "beating", "kicking"],
    "shoving": ["pushing", "shoving"],
    "falling": ["falling", "tripping", "stumbling", "slipping", "losing balance", "falling down"],
    "shouting": ["shouting", "yelling", "screaming"]
}

# Tokenize Primary Labels
primary_labels = []
label_map = {}
for category, descriptions in primary_classes.items():
    for desc in descriptions:
        primary_labels.append(desc)
        label_map[desc] = category

primary_text = open_clip.tokenize(primary_labels).to(device)

# Start Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Unable to access the camera")
    exit()

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Failed to capture frame")
        break
    
    frame_count += 1
    if frame_count % 3 != 0:  # Process every 3rd frame for efficiency
        cv2.imshow("Violence Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue
    
    # Convert frame to PIL Image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        text_features = model.encode_text(primary_text)

    # Compute similarity scores
    similarity = (image_features @ text_features.T).softmax(dim=-1)
    best_description = primary_labels[similarity.argmax()]
    best_category = label_map[best_description]
    best_score = similarity.max().item()

    # Primary class detection
    thresholds = {"normal": 0.80, "running": 0.75, "violence": 0.85}
    detected_class = "normal"
    if best_score >= thresholds.get(best_category, 0):
        detected_class = best_category

    action_text = detected_class

    # If violence is detected, check for secondary actions
    if detected_class == "violence":
        action_scores = {}
        for action, descriptions in secondary_classes.items():
            action_texts = open_clip.tokenize(descriptions).to(device)
            action_features = model.encode_text(action_texts)
            similarity_action = (image_features @ action_features.T).softmax(dim=-1)
            action_scores[action] = similarity_action.max().item()

        best_action = max(action_scores, key=action_scores.get)
        action_confidence = action_scores[best_action]
        if action_confidence >= 0.65:
            action_text = best_action

    # Draw label on frame
    label_text = f"{detected_class.upper()} - {action_text} ({best_score:.2f})"
    color = (0, 255, 0) if detected_class == "normal" else (0, 0, 255) if detected_class == "violence" else (255, 165, 0)
    
    cv2.putText(frame, label_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    cv2.imshow("Violence Detection", frame)
    print(f"📢 Camera → Detected: {detected_class}, {action_text} | Score: {best_score:.2f}")

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🎉 Real-time detection stopped.")
