import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import cv2
import numpy as np
from collections import defaultdict, deque
import supervision as sv
from ultralytics import YOLO

# Setup the Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['PROCESSED_FOLDER'] = 'processed/'

# Ensure the folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

# Source polygon zone
SOURCE = np.array([[1252, 787], [2298, 803], [5039, 2159], [-550, 2159]])

TARGET_WIDTH = 25
TARGET_HEIGHT = 250
TARGET = np.array(
    [
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_HEIGHT - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1]
    ]
)


class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return redirect(url_for('index'))

    file = request.files['video']
    if file.filename == '':
        return redirect(url_for('index'))

    # Save the uploaded video
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(video_path)

    # Set the path for the processed video
    processed_video_path = os.path.join(app.config['PROCESSED_FOLDER'], f"processed_{file.filename}")

    # Process the video
    process_video(video_path, processed_video_path)

    return redirect(url_for('download_file', filename=f"processed_{file.filename}"))


@app.route('/processed/<filename>')
def download_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)


def process_video(source_video_path: str, target_video_path: str):
    video_info = sv.VideoInfo.from_video_path(source_video_path)
    model = YOLO("yolov8s.pt")
    byte_track = sv.ByteTrack(frame_rate=video_info.fps)
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=video_info.resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)

    bounding_box_annotator = sv.BoxCornerAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        text_scale=text_scale,
        text_thickness=thickness,
        text_position=sv.Position.BOTTOM_CENTER
    )

    frame_generator = sv.get_video_frames_generator(source_video_path)
    polygon_zone = sv.PolygonZone(SOURCE, frame_resolution_wh=video_info.resolution_wh)
    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

    coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))

    with sv.VideoSink(target_video_path, video_info) as sink:
        for frame in frame_generator:
            result = model(frame)[0]
            detections = sv.Detections.from_ultralytics(result)
            detections = detections[polygon_zone.trigger(detections)]
            detections = byte_track.update_with_detections(detections=detections)

            points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            points = view_transformer.transform_points(points=points).astype(int)

            labels = []
            for tracker_id, [_, y] in zip(detections.tracker_id, points):
                coordinates[tracker_id].append(y)
                if len(coordinates[tracker_id]) < video_info.fps / 2:
                    labels.append(f"{tracker_id}")
                else:
                    coordinate_start = coordinates[tracker_id][-1]
                    coordinate_end = coordinates[tracker_id][0]
                    distance = abs(coordinate_start - coordinate_end)
                    time = len(coordinates[tracker_id]) / video_info.fps
                    speed = distance / time * 3.6
                    labels.append(f"#{tracker_id} {int(speed)} km/h")

            annotated_frame = frame.copy()
            annotated_frame = sv.draw_polygon(annotated_frame, polygon=SOURCE, color=sv.Color.RED)
            annotated_frame = bounding_box_annotator.annotate(scene=annotated_frame, detections=detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
            sink.write_frame(annotated_frame)

    return target_video_path


if __name__ == '__main__':
    app.run(debug=True)
