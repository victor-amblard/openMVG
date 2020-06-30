#ifndef OPENMVG_SFM_SFM_GEOMETRIC_FEATURES_EXTRACTION_HPP
#define OPENMVG_SFM_SFM_GEOMETRIC_FEATURES_EXTRACTION_HPP



#include "openMVG/multiview/projection.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/opencv.hpp>
#define PCL_NO_PRECOMPILE

#include <pcl/point_types.h>
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

#include <ceres/rotation.h>
#include <ceres/types.h>

#include "openMVG/sfm/sfm_line_utils.hpp"
#include "openMVG/sfm/sfm_data.hpp"
#include "openMVG/numeric/eigen_alias_definition.hpp"

namespace openMVG {
namespace sfm {
using PointCloudXYZ = pcl::PointCloud<pcl::PointXYZ>;

/** 
 * Loads lines outputted by a MyLine Segment Detector (LSD)
**/
void loadLinesFromLSDFile(const std::string& filename,
                          const std::string& imgFilename,
                          std::vector<Endpoints2>& allLinesImage);

/**
 * Processes the lines extracted by loadLinesFromLSDFile 
**/
void processLinesLSD(const std::string& imgFilename,
                     std::vector<Endpoints2>& allLinesImage);

std::vector<Endpoints2> getLinesInImage(cv::Mat & img, const bool& visualization=false);

std::vector<std::pair<int, bool>> getViewsSegment(const Segment3D& segment,
                                                  const SfM_Data& sfm_data,
                                                  const Mat3& K);
/**
 * Takes a reference segment (refSegment) and current segment (curSegment)
 * and evaluates whether those two segments are two different views of the same 3D line
 * The criteria are 
 *    - distance of reprojected endpoints from curSegment to refSegment
 *    - Angular difference between the directions of the reprojected refSegment and the 2D curSegment
 *    - length of the overlap between the two segments
**/ 
bool isMatched(const Segment3D& curSegment,
              const Segment3D& refSegment,
              const Pose3& transformWF,
              const int w,
              const int h,
              bool completeVisibility,
              const Mat3& K);

/**
 * Main function: Evaluates all potential data association across the dataset
 * Segments are sorted according to their length (decreasing order) and then each segment
 * is being tested against all the other segments (using isMatched functionn)
 * A union-find approach is used to determine clusters of segment
**/
void findCorrespondencesAcrossViews(const std::vector<std::string>& filenames,
                                    std::vector<std::pair<int, Segment3D>>& allSegments,
                                    const std::vector<std::vector<int>>& segmentsInView,
                                    const SfM_Data& sfm_data,
                                    const Mat3& K,
                                    std::vector<std::vector<int>>&finalLines,
                                    std::map<int, int>& mapSegment);


void visualize3dLines(const std::vector<std::pair<MyLine, std::vector<int>>>* lines,
                      PointCloudXYZ::Ptr& cloud,
                      std::vector<Eigen::Vector6f>& intersections);

void visualizeMatches(const std::vector<std::pair<IndexT, std::vector<IndexT>>>& matches,
                      const Hash_Map<IndexT, Eigen::Vector6d>& all_3d_lines,
                      const Hash_Map<IndexT, std::pair<IndexT, Eigen::Vector4d>>& all_2d_lines,
                      const Hash_Map<IndexT, Hash_Map<IndexT, Eigen::Vector4d>>& proj_3d_lines,
                      const View * v,
                      const std::string& rootPath);

void testLineReprojectionCostFunction(const double * const cam_intrinsics,
                                      const double * const cam_extrinsics,
                                      const double * const line_3d_endpoint,
                                      const double * m_line_2d_endpoints,
                                      const View * v);

void testLineReprojectionPlucker(const double * const cam_intrinsics,
                                      const double * const cam_extrinsics,
                                      const double * const line_3d_endpoint,
                                      const double * m_line_2d_endpoints,
                                      const View * v);
}
}
#endif