import torch
from torchvision import transforms
from PIL import Image
import streamlit as st
import torch.nn as nn
import torch.nn.functional as F

# Define the CNN model
class PlantDiseaseModel(nn.Module):
    def __init__(self, num_classes):
        super(PlantDiseaseModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, padding=1, kernel_size=3, stride=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, padding=1, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, padding=1, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(128, 256, padding=1, kernel_size=3, stride=1)
        
        # Adjust dimensions for fully connected layer
        self.fc1 = nn.Linear(256 * 16 * 16, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        
        # Flatten the tensor before passing to fully connected layers
        x = x.view(-1, 256 * 16 * 16)  
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load model checkpoint
checkpoint = torch.load("best_model.pt", map_location=torch.device('cpu'))
model = PlantDiseaseModel(num_classes=38)
model.load_state_dict(checkpoint['model_state'])
model.eval()

# Class labels
class_names = {
    0: "Apple___Apple_scab",
    1: "Apple___Black_rot",
    2: "Apple___Cedar_apple_rust",
    3: "Apple___healthy",
    4: "Blueberry___healthy",
    5: "Cherry_(including_sour)___Powdery_mildew",
    6: "Cherry_(including_sour)___healthy",
    7: "Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot",
    8: "Corn_(maize)___Common_rust_",
    9: "Corn_(maize)___Northern_Leaf_Blight",
    10: "Corn_(maize)___healthy",
    11: "Grape___Black_rot",
    12: "Grape___Esca_(Black_Measles)",
    13: "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    14: "Grape___healthy",
    15: "Orange___Haunglongbing_(Citrus_greening)",
    16: "Peach___Bacterial_spot",
    17: "Peach___healthy",
    18: "Pepper,_bell___Bacterial_spot",
    19: "Pepper,_bell___healthy",
    20: "Potato___Early_blight",
    21: "Potato___Late_blight",
    22: "Potato___healthy",
    23: "Raspberry___healthy",
    24: "Soybean___healthy",
    25: "Squash___Powdery_mildew",
    26: "Strawberry___Leaf_scorch",
    27: "Strawberry___healthy",
    28: "Tomato___Bacterial_spot",
    29: "Tomato___Early_blight",
    30: "Tomato___Late_blight",
    31: "Tomato___Leaf_Mold",
    32: "Tomato___Septoria_leaf_spot",
    33: "Tomato___Spider_mites_Two-spotted_spider_mite",
    34: "Tomato___Target_Spot",
    35: "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    36: "Tomato___Tomato_mosaic_virus",
    37: "Tomato___healthy"
}


# Preprocess the image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

# Prediction function
def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = output.argmax(1).item()
        probabilities = torch.nn.functional.softmax(output[0], dim=0).numpy()
    return predicted_class, probabilities

# Extract plant and disease information from the class name
def extract_plant_and_disease(class_name):
    plant_type, disease_type = class_name.split("___")
    return plant_type, disease_type

# Streamlit UI
st.title("Plant Disease Detection")
st.write("This web app uses machine learning to classify plant leaf diseases. Upload an image of a plant leaf, and it will predict the disease.")

# Image upload
uploaded_file = st.file_uploader("Upload an image of the plant leaf", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Prediction and analysis
    with st.spinner('Analyzing...'):
        predicted_class, probabilities = predict(uploaded_file)

    # Get plant type and disease type
    predicted_class_name = class_names[predicted_class]
    plant_type, disease_type = extract_plant_and_disease(predicted_class_name)
    
    st.success(f"Prediction: {predicted_class_name}")
    st.write(f"### Plant Type: {plant_type}")
    st.write(f"### Disease Type: {disease_type}")

    # Displaying probabilities for the bar chart
    prob_df = {class_names[i]: prob for i, prob in enumerate(probabilities)}
    st.subheader("Prediction Confidence:")
    st.bar_chart(prob_df)
    
    # Show top 3 most likely diseases
    sorted_probs = sorted(prob_df.items(), key=lambda x: x[1], reverse=True)
    top_3 = sorted_probs[:3]
    
    st.write("### Top 3 Disease Predictions:")
    for i, (disease, prob) in enumerate(top_3):
        st.write(f"{i+1}. {disease} - Probability: {prob*100:.2f}%")

    # Additional tips (optional)
    st.write("### Care Tips for the Plant:")
    st.write("Make sure to regularly inspect the plant for signs of disease. Follow best practices for preventing fungal and bacterial infections.")

# Enhancing Streamlit App with Customization
st.sidebar.header("About")
st.sidebar.write(
    "This app is developed for detecting plant diseases using deep learning models. You can upload an image and get the disease prediction along with confidence scores."
)

# Footer
st.markdown(
    """
    ---
    Developed by Saketh Komirishetty
    """
)
