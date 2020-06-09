#include "line_utils.hpp"
#include "openMVG/sfm/sfm_data.hpp"
#include "openMVG/multiview/projection.hpp"
#include <opencv2/opencv.hpp>

#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/common/intersections.h>
#include <pcl/common/common_headers.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>

#include <pcl/filters/extract_indices.h>
#include <pcl/kdtree/kdtree_flann.h>



#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>

#include <pcl/ModelCoefficients.h>




namespace openMVG {
namespace sfm {
using PointCloudXYZ = pcl::PointCloud<pcl::PointXYZ>;

/**
 * Extract planes from a point cloud 
*/
void extractPlanesFromCloud(PointCloudXYZ::Ptr filteredCloud,
                            std::vector<pcl::ModelCoefficients>& planes,
                            std::vector<PointCloudXYZ::Ptr, Eigen::aligned_allocator<PointCloudXYZ::Ptr> >& outputClouds);
/**
 *  Takes LIDAR detected planes and outputs all possible lines based on plane intersections
*/ 
void findLinesFromPlanes(const std::vector<pcl::ModelCoefficients>& planesCoeff,
                        std::vector<Eigen::Vector6f>& linesCoeff);

/** 
 * Identify actual lines from the set of all potential lines using the method described in
 * Edge Detection and Feature Line Tracing in 3D-Point Clouds by Analyzing Geometric Properties of Neighborhoods 
*/
void checkInlierLines(const std::vector<Eigen::Vector6f>& allLines,
                      PointCloudXYZ::Ptr pCloud,
                      std::vector<std::pair<Line, std::vector<int>>> * results);


/**
 * Compute angular gap  = max theta_{i+1} - theta{i}
 * Utility function for checkInlierLines
*/ 
float computeAngularGap(PointCloudXYZ::Ptr inlierCloud, 
                        const pcl::ModelCoefficients& planeCoeff, 
                        const pcl::PointXYZ& poi);

/**
 * Further refines line estimates using a RANSAC approach to identify inlier points belonging
 * to the previously detected lines
*/
void checkValidPointsOnLines(const std::vector<Eigen::Matrix<float,6,1>>& allLines,
                            PointCloudXYZ::Ptr pCloud,
                            std::vector<std::pair<int, int>>& pointsOnLine,
                            std::vector<std::pair<Line, std::vector<int>>> * results);


/**
 * Line detection in an image using Canny edge detector and Probabilistic Hough transform
*/
void getLinesInImageCanny(const std::string& filename,
                          std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>>& allLines,
                          const bool& visualization=false);


/**
 * Same function as above but instead of using a Canny edge detector then Hough transform
 * we are using a neural network with attraction fields cf. Learning Attraction Field Representation for Robust Line Segment Detection, CVPR '19
 * It yields fewer lines than the above approach and thus is more robust for 2D/3D line correspondence
*/
void getLinesInImageAfm(const std::string& filename,
                        std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>>& allLines,
                        const bool& visualization=false);

void loadAllAfMFiles(const std::vector<std::string>& filenames,
                     std::vector<std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>>>& allLines);

/**
 * Robustly matches 3D lines from SfM point cloud and 2D lines detected with getLinesInImage
 * @return a pair, first element represents the index of the match and second element represents the confidence on the match
*/
std::pair<int,float> matchLine2Line(const std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>>& detectedLines,
                                    const Eigen::Vector3f backProjectedLine);


int getLineLineCorrespondence(const Eigen::Vector4d& cur2dSegment,
                              const std::vector<std::pair<IndexT, Eigen::Vector4d>> allVisible3dlines,
                              const Mat3& K,
                              const geometry::Pose3& pose);

/**
 * Takes the vector of all 3D lines and returns all the lines visible in the current view
 * This excludes:
 *  a) All the lines that are not in the FOV
 *  b) All the lines in FOV but occluded
 * There is no easy way to know whether a line is occluded or not
 * The heuristic consists in looking at all the image features observed in the current view and in the vicinity of the 3D line and 
 * discard all lines for which no visual feature is found
*/ 
std::vector<std::pair<IndexT, Eigen::Vector4d>> getAllVisibleLines(const Hash_Map<IndexT, Eigen::Vector6d>& all_3d_lines,
                                       const geometry::Pose3& pose,
                                    const Mat3& intr,
                                    const View * view);


bool isLineInFOV(const Eigen::Vector6d& line,
                 const IndexT width, 
                 const IndexT height,
                 const Mat3& K,
                 const geometry::Pose3& pose,
                 Eigen::Vector4d& endPoints);


//TODO
bool isLineOccluded(void);

void visualize3dLines(const std::vector<std::pair<Line, std::vector<int>>>* lines,
                      PointCloudXYZ::Ptr& cloud,
                      std::vector<Eigen::Vector6f>& intersections);

void visualizeMatches(const std::vector<std::pair<IndexT, std::vector<IndexT>>>& matches,
                      const Hash_Map<IndexT, Eigen::Vector6d>& all_3d_lines,
                      const Hash_Map<IndexT, std::pair<IndexT, Eigen::Vector4d>>& all_2d_lines,
                      const Hash_Map<IndexT, Hash_Map<IndexT, Eigen::Vector4d>>& proj_3d_lines,
                      const View * v,
                      const std::string& rootPath);
}
}