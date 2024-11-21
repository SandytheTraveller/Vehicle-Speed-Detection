import argparse
import cv2

import numpy as np

# computer vision library
import supervision as sv
from inference.models.utils import get_roboflow_model

# the list of vertices of the polygon zone
SOURCE = np.array([[1252, 787],
                   [2298, 803],
                   [5039, 2159],
                   [-550, 2159]])


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

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    video_info = sv.VideoInfo.from_video_path(args.source_video_path)

    # 640 - input resolution
    model = get_roboflow_model("yolov8x-640")

    # adding the tracking to the system
    byte_track = sv.ByteTrack(frame_rate=video_info.fps)

    thickness = sv.calculate_optimal_line_thickness(
        resolution_wh=video_info.resolution_wh
    )
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)
    # bounding box annotation
    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=thickness)
    # labels annotation
    label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_thickness=thickness)

    # a generator that returns the frames of the video
    # we use it to loop over the frames of the video
    frame_generator = sv.get_video_frames_generator(args.source_video_path)

    polygon_zone = sv.PolygonZone(SOURCE, frame_resolution_wh=video_info.resolution_wh)

    for frame in frame_generator:
        # run inference on every frame
        result = model.infer(frame)[0]
        # convert results into supervision Detection object
        detections = sv.Detections.from_inference(result)
        detections = detections[polygon_zone.trigger(detections)]
        detections = byte_track.update_with_detections(detections=detections)

        labels = [
            f"#{tracker_id}"
            for tracker_id in detections.tracker_id
        ]
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

        cv2.imshow('Vehicle Speed Detection', annotated_frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()
