#include "detection_display.h"

#include <opencv2/imgproc.hpp>

std::shared_ptr<DetectionDisplay> DetectionDisplay::build(
  const std::string& displayName, 
  std::vector<std::string> labelMap) {

  displayName_ = displayName;
  labelMap_ = labelMap;
  return std::static_pointer_cast<DetectionDisplay>(shared_from_this());
}

std::shared_ptr<dai::Buffer> DetectionDisplay::processGroup(
  std::shared_ptr<dai::MessageGroup> in) {

  auto baseFramePtr = in->get<dai::ImgFrame>("images");
  auto detectionsPtr = in->get<dai::SpatialImgDetections>("detections");
  
  cv::Mat cvFrame = baseFramePtr->getCvFrame();
  for (const dai::SpatialImgDetection detection : detectionsPtr->detections) {
    drawDetection(cvFrame, detection);
  }

  runDisplayWindow(cvFrame);

  // This is a terminal node, no data needs to be processed
  return nullptr;
}

void DetectionDisplay::drawDetection(cv::Mat& frame, const dai::SpatialImgDetection& detection) {
  cv::Scalar colorGreen(0, 255, 0);
  cv::Scalar colorWhite(255, 255, 255);
  cv::Scalar colorRed(0, 0, 255);

  // -- Draw our bounding box -- 
  // Ensure the frame's coordinates are accurate to our output frame
  int width = frame.cols;
  int height = frame.rows;
  int xmax = width * detection.xmax;
  int xmin = width * detection.xmin;
  int ymax = height * detection.ymax;
  int ymin = height * detection.ymin;

  // -- Render our label and coordinates -- 
  std::string label;
  // The labelMap has no value for the class type (shouldn't happen)
  try {
    label = labelMap_[detection.label];
  } catch(...) {
    label = std::to_string(detection.label);
  }

  cv::Scalar rectColor = colorGreen;
  if (detection.spatialCoordinates.z < 300) {
    label = label + " [TOO CLOSE]";
    rectColor = colorRed;
  }

  cv::rectangle(frame, cv::Point(xmax, ymax), cv::Point(xmin, ymin),
                rectColor, 1);
  cv::putText(frame, label, cv::Point(xmin, ymin-5), 
              cv::FONT_HERSHEY_TRIPLEX, 0.5, colorWhite);

  std::string xStr = std::to_string((int) detection.spatialCoordinates.x);
  std::string yStr = std::to_string((int) detection.spatialCoordinates.y);
  std::string zStr = std::to_string((int) detection.spatialCoordinates.z);
  cv::putText(frame, "X: " + xStr + "mm", cv::Point(xmin+3, ymin+15), 
              cv::FONT_HERSHEY_TRIPLEX, 0.5, colorWhite);
  cv::putText(frame, "Y: " + yStr + "mm", cv::Point(xmin+3, ymin+30),
              cv::FONT_HERSHEY_TRIPLEX, 0.5, colorWhite);
  cv::putText(frame, "Z (Depth): " + zStr + "mm", cv::Point(xmin+3, ymin+45), 
              cv::FONT_HERSHEY_TRIPLEX, 0.5, colorWhite);
}

void DetectionDisplay::runDisplayWindow(cv::Mat& frame) {
  cv::imshow(displayName_, frame);

  // Wait for 1 ms to give the window time to update before checking
  // for key presses (was set to 0 but that blocked updates)
  if (cv::waitKey(1) == 'q') {
    // Stops the pipeline. Weird placement; depthai convention.
    // Allows our user to quit the app within the display window
    stopPipeline();  
  }
}

 

