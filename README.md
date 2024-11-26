# Vehicle-Speed-Detection

Speed Estimation Wapplication for Vehicles
This is the repository that contains my web application which calculates the speed of vehicles in the footage.
This is not a complete project, there is a room for improvement.
So far, here's what's been done:

- Video Upload & Processing: Users can upload videos via a web interface. The application processes these videos to detect vehicle speed.
- YOLOv8 for Detection: The YOLOv8 model is used to detect vehicles in the video frames.
- Speed Calculation: The system calculates vehicle speed using a perspective transformation and tracks movements across frames.
- Visualization: The app annotates vehicles with their speed on the processed video and allows users to download the results.

To do next:
- Improve speed calculation by refining the perspective transformation and vehicle tracking over time.
- Optimize the application for real-time video processing and detection.
- Add progress bars or visual feedback for video uploads and processing.
- Experiment with different YOLO versions or other object detection models for enhanced performance.
