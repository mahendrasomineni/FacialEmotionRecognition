import cv2
from PIL import Image
from models import *
class_dict={0:'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print("Model loading...")
model = torch.load('encoder_decoder_model.pth',map_location=torch.device('cpu'))
print("Done loading")
model.eval()

# Define a transformation to preprocess the face region before passing it to the model
data_transform = transforms.Compose([
    # transforms.Resize((45, 45)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Initialize video capture (0 is for the default webcam, or provide the video file path)
video_capture = cv2.VideoCapture(0)  # Change to the appropriate video source

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Iterate through detected faces
    for (x, y, w, h) in faces:
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Extract the face region for classification
        face_roi = frame[y:y + h, x:x + w]

        # Resize the face region to match the required input size for the model
        face_resized = cv2.resize(face_roi, (48, 48))

        # Convert the resized face region to PIL Image
        pil_image = Image.fromarray(face_resized)
        input_tensor = data_transform(pil_image).unsqueeze(0)

        #Model prediction
        with torch.no_grad():
            output = model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()

        # Display the classification result (predicted_class) on the frame
        # Example:
        cv2.putText(frame, f'Class: {class_dict[int(predicted_class)]}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
video_capture.release()
cv2.destroyAllWindows()