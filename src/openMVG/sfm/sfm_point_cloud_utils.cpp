#include "openMVG/sfm/sfm_point_cloud_utils.hpp"

namespace openMVG{
namespace sfm{

Eigen::Matrix4d convertRTEigen(const geometry::Pose3& pose)
{
    //Todo    
    const Mat3 R = pose.rotation();
    const Vec3 t = pose.translation();

    Eigen::Matrix4d result = Eigen::Matrix4d::Identity();
    result.block(0,0,3,3) = R;
    result.block(0,3,3,1) = t;
    return result;
}

inline bool readPointCloud(const std::string& filename, 
                           PointCloudXYZ::Ptr result)
{
    int success = pcl::io::loadPCDFile<pcl::PointXYZ>(filename, *result);
    return (success != -1);
}

void visualizePointCloud(PointCloudXYZ::Ptr pointCloud)
{
    pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0, 0, 0);
      viewer->addPointCloud<pcl::PointXYZ> (pointCloud, "sample cloud");

    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
    viewer->addCoordinateSystem (1.0);
    viewer->initCameraParameters ();

   while (!viewer->wasStopped ())
    {
      viewer->spinOnce (100);
    };
}
std::vector<PointCloudXYZ::Ptr> readAllClouds(const std::string& dirName, 
                                              const std::vector<std::string>& allFilenames)
{
    std::vector<PointCloudXYZ::Ptr> results;

    for (std::string filename: allFilenames){
        PointCloudXYZ::Ptr curCloud(new PointCloudXYZ);
        //WARNING !!!
        std::string xFilename = dirName + "/" + filename.substr(0,4)+".pcd";

        if (!readPointCloud(xFilename, curCloud))
            results.push_back(nullptr);
        else
            results.push_back(curCloud);
    }

    return results;
}

bool fusePointClouds(const std::vector<PointCloudXYZ::Ptr>& allClouds,
                     const Poses& poses,
                     const Eigen::Matrix4d& lid2cam,
                     PointCloudXYZ::Ptr fusedPcl,
                     const float leafSize)
{

    PointCloudXYZ::Ptr globalCloud(new PointCloudXYZ);
    size_t N = allClouds.size();

    if (poses.size()!=N)
        return false;


    for (const auto& pose_it : poses){

        const IndexT indexPose = pose_it.first;

        const geometry::Pose3& pose = pose_it.second;
        Eigen::Matrix4d curPose = convertRTEigen(pose)*lid2cam;
        std::cerr << curPose << std::endl;


        std::stringstream ss;
        PointCloudXYZ::Ptr curCloud = allClouds.at(indexPose);
        PointCloudXYZ::Ptr curTfCloud(new PointCloudXYZ);

        pcl::transformPointCloud(*curCloud, *curTfCloud, curPose);

        for(auto it=curTfCloud->begin();it!=curTfCloud->end();++it)
            globalCloud->push_back(*it);
        
    }

    pcl::octree::OctreePointCloudDensity<pcl::PointXYZ> octreeB(leafSize);
    octreeB.setInputCloud(globalCloud);
    octreeB.addPointsFromInputCloud ();
    typedef std::vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ> > AlignedPointTVector;
    AlignedPointTVector voxelCenters;
    voxelCenters.clear();

    auto allVoxels = octreeB.getOccupiedVoxelCenters(voxelCenters);

    for(auto it = voxelCenters.begin();it!=voxelCenters.end();++it)
        fusedPcl->push_back(*it);
    
    return true;
}
} // namespace sfm
} // namespace openMVG