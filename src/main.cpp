#include <memory> // For shared pointers
#include <iostream>
#include <string>
#include <vector>

#include <cxxopts.hpp>
#include <opencv2/opencv.hpp>
#include <depthai/depthai.hpp>

#include "detection_display.h"

// This code is designed to run with OAK-D Series 2 camera
constexpr dai::CameraBoardSocket COLOR_SOCKET = dai::CameraBoardSocket::CAM_A;
constexpr dai::CameraBoardSocket LEFT_SOCKET = dai::CameraBoardSocket::CAM_B;
constexpr dai::CameraBoardSocket RIGHT_SOCKET = dai::CameraBoardSocket::CAM_C;

int main(int argc, char** argv) {
  // -- Parse command line arguments --
  cxxopts::Options options("ObjectDetector", 
    "Runs a multi-node pipeline for visual object and depth detection");

  options.add_options()
    ("w,width", "Width of the camera output in pixels; must be a multiple of 16",
     cxxopts::value<uint32_t>()->default_value("640"))
    ("h,height", "Height of the camera output in pixels; must be a multiple of 16",
     cxxopts::value<uint32_t>()->default_value("480"))
    ("m,model", "String value for the name of the object detection neural network", 
     cxxopts::value<std::string>()->default_value("yolov6-nano"))
    ("b,box-scale", "Sets the scale of the bounding box used for depth calculations", 
     cxxopts::value<float>()->default_value("0.5"))
    ("l,l-threshold,lower-threshold", "Pixel depth values below this number will be ignored in depth calculation",
     cxxopts::value<uint32_t>()->default_value("100"))
    ("u,u-threshold,upper-threshold", "Pixel depth values above this number will be ignored in depth calculation", 
     cxxopts::value<uint32_t>()->default_value("5000"))
    ("a,alg,algorithm", "The algorithm used to calculate object depth. average/mean/min/max/mode/median", 
     cxxopts::value<std::string>()->default_value("average"))
    ("s,step-size", "The distance between pixels that are used in depth calculation", 
     cxxopts::value<int>()->default_value("1")) 
    ("f,fps", "The amount of frames per second that are processed",
     cxxopts::value<float>()->default_value("30.0"));
  cxxopts::ParseResult result = options.parse(argc, argv);

  // -- Set up depthai pipeline --
  dai::Pipeline pipeline;
  auto colorCamPtr = pipeline.create<dai::node::Camera>()->build(COLOR_SOCKET);
  auto leftCamPtr = pipeline.create<dai::node::Camera>()->build(LEFT_SOCKET);
  auto rightCamPtr = pipeline.create<dai::node::Camera>()->build(RIGHT_SOCKET);

  // We start to have some issues processing output above this size
  uint32_t width = result["width"].as<uint32_t>();
  uint32_t height = result["height"].as<uint32_t>();
  const std::pair<uint32_t, uint32_t> outputSize = std::make_pair(width, height);
  dai::Node::Output* leftOut = leftCamPtr->requestOutput(outputSize);
  dai::Node::Output* rightOut = rightCamPtr->requestOutput(outputSize);

  auto stereoProcessorPtr = pipeline.create<dai::node::StereoDepth>();
  stereoProcessorPtr->setOutputSize(width, height);
  stereoProcessorPtr->setExtendedDisparity(true); // Important for short range objects

  // This takes up a shave on the camera and as a result we need a NN compiled for 7 shaves
  // we may want to do this eventually, but it will end up lowering our fps
  // stereoProcessorPtr->setDefaultProfilePreset(dai::node::StereoDepth::PresetMode::ROBOTICS); 
  leftOut->link(stereoProcessorPtr->left);
  rightOut->link(stereoProcessorPtr->right);

  // Model should download from the luxonis model zoo onto the Camera device
  dai::NNModelDescription modelDesc;
  modelDesc.model = result["model"].as<std::string>();

  float scale = result["box-scale"].as<float>();
  uint32_t upperThresh = result["upper-threshold"].as<uint32_t>();
  uint32_t lowerThresh = result["lower-threshold"].as<uint32_t>();
  int stepSize = result["step-size"].as<int>();
  float fps = result["fps"].as<float>();

  auto spatialDetectionNtwrkPtr = pipeline.create<dai::node::SpatialDetectionNetwork>();
   // Prevents freezing; older frames are pushed out of full queue (data loss)
  spatialDetectionNtwrkPtr->input.setBlocking(false); 
  // Shrink the bounding box to make depth data more reliable by removing 
  // some background from the object
  spatialDetectionNtwrkPtr->setBoundingBoxScaleFactor(scale);
  // We assume that values reported too close or too far are inaccurate, and 
  // shouldn't be considered for object detection.
  spatialDetectionNtwrkPtr->setDepthLowerThreshold(lowerThresh);
  spatialDetectionNtwrkPtr->setDepthUpperThreshold(upperThresh);
  spatialDetectionNtwrkPtr->setSpatialCalculationStepSize(stepSize);

  std::string algoArg = result["algorithm"].as<std::string>();
  auto algorithm = dai::SpatialLocationCalculatorAlgorithm::AVERAGE;
  if (algoArg == "min") {
    algorithm = dai::SpatialLocationCalculatorAlgorithm::MIN;
  } else if (algoArg == "max") {
    algorithm = dai::SpatialLocationCalculatorAlgorithm::MAX;
  } else if (algoArg == "mode") {
    algorithm = dai::SpatialLocationCalculatorAlgorithm::MODE;
  } else if (algoArg == "median") {
    algorithm = dai::SpatialLocationCalculatorAlgorithm::MEDIAN;
  } else if (algoArg != "mean" && algoArg != "average") {
    printf("Unrecognized algorithm arg: '%s' using average algorithm", algoArg.c_str());
  }
  spatialDetectionNtwrkPtr->setSpatialCalculationAlgorithm(algorithm);
  spatialDetectionNtwrkPtr->build(colorCamPtr, stereoProcessorPtr, modelDesc, fps);

  auto detectionDisplayPtr = pipeline.create<DetectionDisplay>();
  spatialDetectionNtwrkPtr->out.link(detectionDisplayPtr->detectionsInput);
  spatialDetectionNtwrkPtr->passthrough.link(detectionDisplayPtr->imagesInput);
  detectionDisplayPtr->build("Display", spatialDetectionNtwrkPtr->getClasses().value_or(std::vector<std::string>()));

  pipeline.start();
  pipeline.wait(); // pipeline can be closed via pressing 'q' in pop-up window.
  return 0;
}