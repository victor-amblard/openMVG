#include "openMVG/sfm/sfm_data.hpp"
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/common/common_headers.h>
#include <pcl/octree/octree.h>
#include <pcl/octree/octree_iterator.h>
#include <pcl/octree/octree_pointcloud.h>
#include <pcl/octree/octree_pointcloud_density.h>
#include <pcl/octree/octree_pointcloud_singlepoint.h>
namespace openMVG{
namespace sfm{
using PointCloudXYZ = pcl::PointCloud<pcl::PointXYZ>;

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
                     PointCloudXYZ::Ptr fusedPcl,
                     const float leafSize=0.03f);


// Utility function to visualize a point cloud
void visualizePointCloud(PointCloudXYZ::Ptr pointCloud);

/**
 * Utility function to convert an openMVG pose representation (R,t) to a 4x4 matrix 
*/
Eigen::Matrix4d convertRTEigen(const geometry::Pose3& pose);

}
}

