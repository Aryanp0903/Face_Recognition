import face_recognition
import cv2
import pickle

# Load encoded face data
with open("encoded_faces.pickle", "rb") as f:
    known_faces = pickle.load(f)

known_encodings = list(known_faces.values())
known_names = list(known_faces.keys())

# Load image for recognition
image_path = "dataset/person1.jpg"
image = cv2.imread(image_path)

# Verify if image is loaded correctly
if image is None:
    raise Exception(f"Error: Unable to load '{image_path}'. Check the file path.")

rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Detect and encode faces
face_locations = face_recognition.face_locations(rgb_image)
face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

# Recognition logic
for face_encoding, face_location in zip(face_encodings, face_locations):
    matches = face_recognition.compare_faces(known_encodings, face_encoding)
    name = "Unknown"

    if True in matches:
        match_index = matches.index(True)
        name = known_names[match_index]

    # Draw rectangle and label
    top, right, bottom, left = face_location
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.putText(image, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

cv2.imshow("Face Recognition", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
