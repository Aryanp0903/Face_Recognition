import face_recognition
import cv2
import os
import pickle

# Encode known faces
def encode_faces():
    encoded_faces = {}
    dataset_path = "dataset"

    for filename in os.listdir(dataset_path):
        image_path = os.path.join(dataset_path, filename)
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)[0]

        # Save encoded data with filename
        encoded_faces[filename.split('.')[0]] = encoding

    # Save encoded data
    with open("encoded_faces.pickle", "wb") as f:
        pickle.dump(encoded_faces, f)

if __name__ == "__main__":
    encode_faces()
    print("Encoding Complete.")
