#ifndef DETECTION_DISPLAY_H_
#define DETECTION_DISPLAY_H_

#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <depthai/depthai.hpp>

/**
 * Host node to display images with visualized object
 * detection information overlayed.
 * 
 * This class must be a host node in order to output
 * frames to a display window on the host machine.
 */
class DetectionDisplay : 
public dai::NodeCRTP<dai::node::HostNode, DetectionDisplay> {

public:
  // References to input endpoints in the HostNode class
  Input& detectionsInput = inputs["detections"];
  Input& imagesInput = inputs["images"];

  /**
   * Builds the DetectionDisplay node with a specified display name and object
   * label map. It is expected that output is linked to the inputs via 
   * Node::Output::link.
   * 
   * @param displayName the unique std::string name of the image display window
   * @param labelMap the neural network provided map of class labels
   * @return a shared pointer to this instance.
   */
  std::shared_ptr<DetectionDisplay> 
  build(const std::string& displayName, std::vector<std::string> labelMap);

  /**
   * Processes a synced group of messages (detection inputs and images).
   * 
   * @return a buffer of data back to the device node
   */
  std::shared_ptr<dai::Buffer> 
  processGroup(std::shared_ptr<dai::MessageGroup> in) override; // Called in host node

private:

  /// the name of the window used to display images
  std::string displayName_;
  /// the different class labels the neural network can generate
  std::vector<std::string> labelMap_;

  /**
   * Draws a bounding box around a detection with coordinates and 
   * the provided class label. 
   */
  void drawDetection(cv::Mat& frame, const dai::SpatialImgDetection& detection);

  /**
   * Manages the window displaying the camera's output on the host machine.
   */
  void runDisplayWindow(cv::Mat& frame);
};

#endif // DETECTION_DISPLAY_H_