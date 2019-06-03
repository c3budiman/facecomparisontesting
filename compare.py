import face_recognition
import math

# Load the jpg files into numpy arrays
biden_image = face_recognition.load_image_file("./images/c3budiman3.jpg")
unknown_image = face_recognition.load_image_file("./images/c3budiman2.jpg")

try:
    biden_face_encoding = face_recognition.face_encodings(biden_image)[0]
    unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]
except IndexError:
    print("I wasn't able to locate any faces in at least one of the images. Check the image files. Aborting...")
    quit()

known_faces = [
    biden_face_encoding
]

# results is an array of True/False telling if the unknown face matched anyone in the known_faces array
results = face_recognition.compare_faces(known_faces, unknown_face_encoding)

face_distances = face_recognition.face_distance(known_faces, unknown_face_encoding)

for i, face_distance in enumerate(face_distances):
    if (face_distance < 0.5):
        koefisien = -0.2
    elif (face_distance <= 0.6 and face_distance > 0.1):
        koefisien = -0.17
    elif (face_distance > 0.6):
        koefisien = 0.05
    else:
        koefisien = 0

    persentasenegatif = face_distance + koefisien
    persentase = math.floor((1 - persentasenegatif) * 100)
    print("Tingkat Kedekatan : {:.2} (lebih kecil lebih baik)".format(face_distance))
    print("Persentase kemiripan : {} % (lebih besar lebih baik)".format(persentase))
    print("- Dengan cutoff < 0.6, apakah orang tersebut pemilik foto? {}".format(face_distance < 0.6))
    print("\n")
