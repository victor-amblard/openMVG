#ifndef OPENMVG_SFM_SFM_POINT_CLOUD_UTILS_HPP
#define OPENMVG_SFM_SFM_POINT_CLOUD_UTILS_HPP



#define PCL_NO_PRECOMPILE
#include <pcl/pcl_macros.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/common/common_headers.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/octree/octree.h>
#include <pcl/octree/octree_iterator.h>
#include <pcl/octree/octree_pointcloud.h>
#include <pcl/octree/octree_pointcloud_density.h>
#include <pcl/octree/octree_pointcloud_singlepoint.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/extract_indices.h>
#include <opencv2/opencv.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "openMVG/sfm/sfm_data.hpp"
#include "openMVG/sfm/sfm_line_utils.hpp"
#include "third_party/lsd/LineDescriptor.hh"

namespace openMVG{
namespace sfm{
using PointCloudXYZ = ::pcl::PointCloud<::pcl::PointXYZ>;

bool readPointCloud(const std::string& filename, 
                           PointCloudXYZ::Ptr result);

bool readPointCloudXYZIRT(const std::string& filename, 
                          pcl::PointCloud<pcl::PointXYZIRT>::Ptr result);
/**
 * Reads PCL point clouds from a vector of filenames 
 * @return vector of Point Cloud pointers
*/
std::vector<PointCloudXYZ::Ptr> readAllClouds(const std::string& dirName, 
                                                       const std::vector<std::string>& allFilenames);

//Takes a vector of point cloud and fuse them into an octomap 
// according to a vector of transforms from lidar frame to world frame
bool fusePointClouds(const std::vector<PointCloudXYZ::Ptr>& allClouds,
                     const Poses& transforms,
                     const Eigen::Matrix4d& lid2cam,
                     PointCloudPtr<pcl::XPointXYZ> fusedPcl,
                     const float leafSize=0.03f);


// Utility function to visualize a point cloud
void visualizePointCloud(PointCloudXYZ::Ptr pointCloud);
void visualizePointCloud(PointCloudPtr<pcl::XPointXYZ> pointCloud);

/**
 * Utility function to convert an openMVG pose representation (R,t) to a 4x4 matrix 
*/
Eigen::Matrix4d convertRTEigen(const geometry::Pose3& pose);

inline double getDistToLine(const Vec3 lPoint, const Vec3 lDir, const Vec3 curPoint);

std::vector<std::pair<double, std::pair<int, int>>> computeEdgeScoreOnLine(const cv::Mat& rangeMap, const bool& visualization=false);

std::vector<std::pair<uint32_t, Vec3>> getPointsInFov(PointCloudPtr<::pcl::PointXYZIRT> allPoints, int width, int height, const Mat3& K);

void edgeScoreToRGB(const PointCloudPtr<::pcl::PointXYZ>& pointCloud, PointCloudPtr<::pcl::XPointXYZ> outputCloud);

std::vector<int> projectPointCloud(const VelodynePointCloud::Ptr inputCloud, 
                                  const bool& visualization,
                                  const cv::Mat& img);

/** 
 * Associates the edges of the current lidar scan in fov to their corresponding lines
**/ 
void associateEdgePoint2Line(const View * v,
                             const std::vector<Endpoints2>& allLines,
                             const VelodynePointCloud::Ptr inputCloud,
                             const cv::Mat& img,
                             const Mat3& K,
                             const geometry::Pose3& transform,
                             std::vector<Segment3D>& result,
                             const Eigen::Matrix4d lidar2camera,
                             const std::vector<std::vector<LBD::Descriptor>>& allDesc);

void visualizeEndResult(PointCloudPtr<pcl::XPointXYZ> mergedCloud, 
                        const std::vector<std::vector<int>>& finalLines,
                        const std::vector<std::pair<int, Segment3D>>& allSegments,
                       std::map<int, int>& mapIdx,
                      const SfM_Data& sfm_data);
}
}

#endif