import os
import shutil

import cv2
import numpy as np
from deepface import DeepFace


def merge_overlapping_periods(period_dict):
    sorted_periods = sorted(period_dict.items(), key=lambda x: x[0][0])

    merged_periods = []
    current_period, current_speaker = sorted_periods[0]

    for next_period, next_speaker in sorted_periods[1:]:
        if current_period[1] >= next_period[0]:
            if current_speaker == next_speaker:
                current_period = (current_period[0], max(current_period[1], next_period[1]))
            else:
                merged_periods.append((current_period, current_speaker))
                current_period, current_speaker = next_period, next_speaker
        else:
            merged_periods.append((current_period, current_speaker))
            current_period, current_speaker = next_period, next_speaker

    merged_periods.append((current_period, current_speaker))
    return dict(merged_periods)


def get_speaker(time_frame, speaker_dict):
    for (start, end), speaker in speaker_dict.items():
        if start <= time_frame <= end:
            return speaker
    return None


def extract_frames(video_path, output_folder, periods, num_frames=50):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)

    if not video.isOpened():
        print("Error: Could not open video.")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for (start_time, end_time), speaker in periods.items():
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        step = 1

        speaker_folder = os.path.join(output_folder, speaker)
        if not os.path.exists(speaker_folder):
            os.makedirs(speaker_folder)

        frame_count = 0
        frame_number = start_frame

        video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        while frame_number < end_frame and frame_count < num_frames:
            success, frame = video.read()

            if not success:
                break

            if frame_count % step == 0:
                frame_filename = os.path.join(speaker_folder, f"{speaker}_frame_{frame_number}.jpg")
                cv2.imwrite(frame_filename, frame)
                print(f"Saved frame {frame_number} for {speaker}")

            frame_number += 1
            frame_count += 1

    video.release()


def detect_and_crop_faces(image_path, face_cascade):
    img = cv2.imread(image_path)

    if img is None:
        print(f"Error reading image: {image_path}")
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        return False

    (x, y, w, h) = faces[0]
    face = img[y:y + h, x:x + w]
    cv2.imwrite(image_path, face)
    return True


def cosine_similarity(embedding1, embedding2):
    embedding1 = np.array(embedding1)
    embedding2 = np.array(embedding2)
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))


def extract_and_save_most_common_face(folder_path, threshold=0.1):
    face_encodings = []
    face_images = {}

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(folder_path, filename)

            try:
                embedding = DeepFace.represent(img_path=file_path, model_name="ArcFace")[0]["embedding"]
                face_encodings.append(embedding)
                face_images[tuple(embedding)] = file_path
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue

    unique_faces = []
    grouped_faces = {}

    for encoding in face_encodings:
        found_match = False
        for unique_face in unique_faces:
            similarity = cosine_similarity(encoding, unique_face)
            if similarity > threshold:
                found_match = True
                grouped_faces[tuple(unique_face)].append(encoding)
                break
        if not found_match:
            unique_faces.append(encoding)
            grouped_faces[tuple(encoding)] = [encoding]

    most_common_group = max(grouped_faces, key=lambda x: len(grouped_faces[x]))
    most_common_image = face_images[most_common_group]

    new_image_path = os.path.join(folder_path, "max_image.jpg")
    shutil.copy(most_common_image, new_image_path)

    print(f"Most common face extracted and saved as {new_image_path}")
    return new_image_path


def get_overlap(range1, range2):
    start1, end1 = range1
    start2, end2 = range2
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    return max(0, overlap_end - overlap_start)
