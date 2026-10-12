#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

#include "pose/YOLO11-POSE.hpp"

int main() {
    // Configuration parameters
    const std::string labelsPath = "../models/coco.names";       // Path to class labels
    const std::string modelPath  = "../models/yolo11n-pose.onnx";
    bool isGPU = true;                                           // Set to false for CPU processing

    YOLO11POSEDetector pose_detector(modelPath, labelsPath, isGPU);

    cv::Mat frame;

    //auto capture = cv::VideoCapture("https://1172509122.rsc.cdn77.org/live/Qms618.stream/chunklist.m3u8");
    //auto capture = cv::VideoCapture("../data/megamind.avi");
    auto capture = cv::VideoCapture(0);
    //auto img = cv::imread("../data/bus.jpg");
    cv::namedWindow("YOLO Algorithm", cv::WINDOW_AUTOSIZE);
    cv::startWindowThread();

    while (capture.read(frame)) {
    // Perform object detection to get bboxs
    std::vector<Detection> poses = pose_detector.detect(frame);

    // Draw bounding boxes on the image
    pose_detector.drawBoundingBox(frame, poses);

    // Display the annotated image


    cv::imshow("YOLO Poses Detection", frame);
    //while (true) {
    int key = cv::waitKey(1) & 0xFF;
        if (key == 'q') {
            break;
        }
    }
    //}

    return 0;
}