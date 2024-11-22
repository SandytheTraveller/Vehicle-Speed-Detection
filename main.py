import argparse
import cv2

import numpy as np
from collections import defaultdict, deque

# computer vision library
import supervision as sv
from ultralytics import YOLO

# the list of vertices of the polygon zone
SOURCE = np.array([[1252, 787],
                   [2298, 803],
                   [5039, 2159],
                   [-550, 2159]])

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
    def __init__(self, source: np.ndarray,target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Vehicle Speed Detection using Inference and Supervision'
    )
    parser.add_argument(
        "-s",
        "--source_video_path",

        required=True,
        help="Path to the source video file",
        type=str,
    )
    parser.add_argument(
        "-t",
        "--target_video_path",
        required=True,
        help='Path to the target video file (output)',
        type=str
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    video_info = sv.VideoInfo.from_video_path(args.source_video_path)

    # 640 - input resolution
    model = YOLO("yolov8s.pt")

    # adding the tracking to the system
    byte_track = sv.ByteTrack(
        frame_rate=video_info.fps
    )

    thickness = sv.calculate_optimal_line_thickness(
        resolution_wh=video_info.resolution_wh
    )
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)

    # bounding box annotation
    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=thickness)

    # labels annotation
    label_annotator = sv.LabelAnnotator(
        text_scale=text_scale,
        text_thickness=thickness,
        text_position=sv.Position.BOTTOM_CENTER
    )
    trace_annotator = sv.TraceAnnotator(
        thickness=thickness,
        trace_length=video_info.fps*2,
        position=sv.Position.BOTTOM_CENTER
    )

    # a generator that returns the frames of the video
    # we use it to loop over the frames of the video
    frame_generator = sv.get_video_frames_generator(args.source_video_path)

    polygon_zone = sv.PolygonZone(SOURCE, frame_resolution_wh=video_info.resolution_wh)
    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

    coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))

    with sv.VideoSink(args.target_video_path, video_info) as sink:
        for frame in frame_generator:
            # run inference on every frame
            result = model(frame)[0]

            # convert results into supervision Detection object
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
            # draw polygone zone on the frame
            # confirm that all unwanted detections are removed
            annotated_frame = sv.draw_polygon(annotated_frame,
                                              polygon=SOURCE,
                                              color=sv.Color.RED)
            annotated_frame = bounding_box_annotator.annotate(
                scene=annotated_frame,
                detections=detections
            )
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame,
                detections=detections,
                labels=labels
            )
            sink.write_frame(annotated_frame)
            cv2.imshow('Vehicle Speed Detection', annotated_frame)
            if cv2.waitKey(1) == ord('q'):
                break

        cv2.destroyAllWindows()
