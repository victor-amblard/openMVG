#include "openMVG/sfm/sfm_point_cloud_utils.hpp"

namespace openMVG{
namespace sfm{

bool readPointCloudXYZIRT(const std::string& filename, 
                           pcl::PointCloud<pcl::PointXYZIRT>::Ptr result)
{
    int sucess = pcl::io::loadPCDFile<pcl::PointXYZIRT>(filename, *result);
    return (sucess != -1);
}
bool readPointCloud(const std::string& filename, 
                           PointCloudXYZ::Ptr result)
{
    int success = pcl::io::loadPCDFile<pcl::PointXYZ>(filename, *result);
    return (success != -1);
}

void visualizePointCloud(PointCloudXYZ::Ptr pointCloud)
{
    ::pcl::visualization::PCLVisualizer::Ptr viewer (new ::pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0, 0, 0);
      viewer->addPointCloud<pcl::PointXYZ> (pointCloud, "sample cloud");

    viewer->setPointCloudRenderingProperties (::pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
    viewer->addCoordinateSystem (1.0);
    viewer->initCameraParameters ();

   while (!viewer->wasStopped ())
    {
      viewer->spinOnce (100);
    };
}
void visualizePointCloud(PointCloudPtr<pcl::XPointXYZ> pointCloud)
{
    ::pcl::visualization::PCLVisualizer::Ptr viewer (new ::pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0, 0, 0);
      viewer->addPointCloud<pcl::XPointXYZ> (pointCloud, "sample cloud");

    viewer->setPointCloudRenderingProperties (::pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
    viewer->addCoordinateSystem (1.0);
    viewer->initCameraParameters ();

   while (!viewer->wasStopped ())
    {
      viewer->spinOnce (100);
    };
}
Vec3 getEndpointLocation(const MyLine& l,
                         const Vec2 pt,
                         const Mat3& K){


    Vec3 result = Vec3::Zero();

    CGAL_K::Point_3 origin((CGAL_K::RT)0, (CGAL_K::RT)0, (CGAL_K::RT)0);
    Vec3 pointCamera = K.inverse() * pt.homogeneous(); //transform image -> camera
    
    CGAL_K::Point_3 endpointProjCamera((CGAL_K::RT)pointCamera(0), (CGAL_K::RT)pointCamera(1), (CGAL_K::RT)pointCamera(2));
    CGAL_K::Line_3 epipolarLine(origin, endpointProjCamera);
    
    CGAL_K::Point_3 cPoint3d((CGAL_K::RT)l.pointDirRep(0), (CGAL_K::RT)l.pointDirRep(1), (CGAL_K::RT)l.pointDirRep(2));
    CGAL_K::Direction_3 cDirLine((CGAL_K::RT)l.pointDirRep(3), (CGAL_K::RT)l.pointDirRep(4), (CGAL_K::RT)l.pointDirRep(5));
    CGAL_K::Line_3 lineEqn(cPoint3d, cDirLine);
    

    if (CGAL::do_intersect(epipolarLine, lineEqn)){
        auto resultInter = CGAL::intersection(epipolarLine, lineEqn);
        if (const CGAL_K::Point_3* p = boost::get<CGAL_K::Point_3>(&*resultInter)){
            result = Vec3(CGAL::to_double(p->x()), CGAL::to_double(p->y()), CGAL::to_double(p->z()));
        }else{
            std::cerr << "Epipolar line is // to 3D line" << std::endl;
        }
    }else{
        //We have 2 skew lines: https://en.wikipedia.org/wiki/Skew_lines
        // v1 = p1 + t*d1, v2 = p2+t*d2
        // n = d1 x d2, n1 = d1 x n
        // c2 = p2 + (p1 - p2).n1/d2.n1 * d2
        CGAL_K::Vector_3 cp = CGAL::cross_product(epipolarLine.to_vector(), lineEqn.to_vector());
        cp = cp/cp.squared_length();
        CGAL_K::Vector_3 normal1 = CGAL::cross_product(epipolarLine.to_vector(), cp);
        CGAL_K::Vector_3 normal2 = CGAL::cross_product(cDirLine.to_vector(), cp);
        CGAL_K::RT coeff =  CGAL::scalar_product(origin - cPoint3d ,normal1) / CGAL::scalar_product(cDirLine.to_vector(), normal1);
        CGAL_K::Point_3 closestPt2Epipolar = cPoint3d + coeff*cDirLine.to_vector();
        result = Vec3(CGAL::to_double(closestPt2Epipolar.x()), CGAL::to_double(closestPt2Epipolar.y()), CGAL::to_double(closestPt2Epipolar.z()));
    }
    return result;
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
                     PointCloudPtr<pcl::XPointXYZ> fusedPcl,
                     const float leafSize)
{

    PointCloudXYZ::Ptr globalCloud(new PointCloudXYZ);
    size_t N = allClouds.size();

    if (poses.size()!=N)
        return false;


    for (const auto& pose_it : poses){

        const IndexT indexPose = pose_it.first;

        const geometry::Pose3& pose = pose_it.second;
        Eigen::Matrix4d curPose = convertRTEigen(pose).inverse()*lid2cam;

        std::stringstream ss;
        PointCloudXYZ::Ptr curCloud = allClouds.at(indexPose);
        PointCloudXYZ::Ptr curTfCloud(new PointCloudXYZ);

        pcl::transformPointCloud(*curCloud, *curTfCloud, curPose);

        for(auto it=curTfCloud->begin();it!=curTfCloud->end();++it)
            globalCloud->push_back(*it);
        
    }

    ::pcl::octree::OctreePointCloudDensity<::pcl::PointXYZ> octreeB(leafSize);
    octreeB.setInputCloud(globalCloud);
    octreeB.addPointsFromInputCloud ();
    typedef std::vector<pcl::PointXYZ, Eigen::aligned_allocator<::pcl::PointXYZ> > AlignedPointTVector;
    AlignedPointTVector voxelCenters;
    voxelCenters.clear();

    auto allVoxels = octreeB.getOccupiedVoxelCenters(voxelCenters);

    for(auto it = voxelCenters.begin();it!=voxelCenters.end();++it){
        pcl::PointXYZ p(it->x, it->y, it->z);
        int nClustered = octreeB.getVoxelDensityAtPoint(p);
        float nC = static_cast<float>(nClustered);
        pcl::XPointXYZ xP;
        xP.x = it->x;
        xP.y = it->y;
        xP.z = it->z;
        xP.nClustered = nC;
        fusedPcl->push_back(xP);
    }
    return true;
}

bool inFOV(const Vec3& p, int w, int h){
    return p(0) < w && p(1) < h && p(1) >= 0 && p(0) >= 0 && p(2) > 0;
}


std::vector<std::pair<uint32_t, Vec3>> getPointsInFov(PointCloudPtr<::pcl::PointXYZIRT> allPoints, int width, int height, const Mat3& K)
{
    std::vector<std::pair<uint32_t, Vec3>> result;
    uint32_t curIt(0);
    for (auto it = allPoints->begin();it!=allPoints->end();++it){
        Vec3 point(it->x, it->y, it->z);
        Vec3 nPoint = projectC2I(point, K);
        if (inFOV(nPoint, width, height))
            result.push_back(std::make_pair(curIt, nPoint));
        ++curIt;
    }
    std::cout << result.size() << " points in fov " << std::endl;
    return result;
}

void associateEdgePoint2Line(const View * v,
                             const std::vector<Endpoints2>& allLines,
                             const VelodynePointCloud::Ptr inputCloud,
                             const cv::Mat& img,
                             const Mat3& K,
                             const geometry::Pose3& pose,
                             std::vector<Segment3D>& result,
                             const Eigen::Matrix4d lidar2camera,
                             const std::vector<std::vector<LBD::Descriptor>>& allDesc)
{

    const int width = v->ui_width;
    const int height = v->ui_height;

    const int colorMapMultiplier = 29;
    const int colorMapLength = 255;

    const size_t sLine(allLines.size());

    for (unsigned int i = 0 ; i < sLine ; ++i)
        cv::line(img, cv::Point(allLines.at(i).first(0),allLines.at(i).first(1)), cv::Point(allLines.at(i).second(0), allLines.at(i).second(1)),
        cv::Scalar(200,200,200));
    std::vector<std::vector<int>> points2lines(sLine);
    std::vector<std::set<int>> linesAdjacency(sLine);

    std::vector<int> edgePoints = projectPointCloud(inputCloud, true, img);
    ::pcl::PointIndices::Ptr edgesIdx(new ::pcl::PointIndices ());

    
    for (auto& edge: edgePoints)
        edgesIdx->indices.push_back(edge);

    VelodynePointCloud::Ptr edgeCloud(new VelodynePointCloud);

    // Extract edge points from lidar point cloud
    ::pcl::ExtractIndices<::pcl::PointXYZIRT> extract;
    extract.setInputCloud (inputCloud);
    extract.setIndices (edgesIdx);
    extract.setNegative (false);
    extract.filter (*edgeCloud);

    pcl::transformPointCloud(*edgeCloud, *edgeCloud, lidar2camera);

    // Get all edge points in fov
    // To change
    std::vector<std::pair<uint32_t, Eigen::Vector3d>> edgesInFov = getPointsInFov(edgeCloud, width, height, K);

    for(size_t jPoint = 0;jPoint<edgesInFov.size();++jPoint){
        Vec2 curPt = edgesInFov.at(jPoint).second.block(0,0,2,1);
        int iMini(-1);
        double miniDist(std::numeric_limits<double>::max());
        std::vector<int> validLines;

        for (size_t i = 0;i<allLines.size();++i){
            auto line = allLines.at(i);
            Vec2 normal = endpoints2Normal(line);
            double curDist = distPoint2Segment(curPt, line, normal);
            double angle2Vert = CLIP_ANGLE(std::acos(normal(1)));

            if (curDist < PARAMS::tDistanceLinePointMatch){
                validLines.push_back(i);
                if (curDist < miniDist && angle2Vert < PARAMS::tVertAngle){ //Since we measure edges vertically it doesn't make sense to have horizontal lines
                    miniDist = curDist;
                    iMini = i;
                }
            }
        }

        if (iMini!=-1){
            points2lines.at(iMini).push_back(jPoint);
            for(auto oLine: validLines){
                if (oLine != iMini){
                    linesAdjacency.at(iMini).insert(oLine);
                    linesAdjacency.at(oLine).insert(iMini);
                }
            }
        }
    }

    std::vector<bool> visibility(edgesInFov.size());
    for(size_t i =0 ; i<visibility.size();++i)
        visibility.at(i) = false;

    int count(0);
    const float threshDistLineRansac(0.1);
    const int maxIterationsLineRansac(50);
    const int minInliersLine(3);

    pcl::SACSegmentation<pcl::PointXYZIRT> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_LINE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(threshDistLineRansac);
    seg.setMaxIterations(maxIterationsLineRansac);

    Eigen::Matrix4d K_mat = Eigen::Matrix4d::Identity();
    K_mat.block(0,0,3,3) = K;
    std::vector<int> finalLines;
    for(size_t iLine=0; iLine<sLine; ++iLine){
        if (points2lines.at(iLine).size() > 2){

            PointCloudPtr<pcl::PointXYZIRT> tmpCloud(new VelodynePointCloud);

            for (auto j=0;j<points2lines.at(iLine).size();++j){
                 tmpCloud->push_back(edgeCloud->points[edgesInFov.at(points2lines.at(iLine).at(j)).first]);
              }

            pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
            pcl::PointIndices::Ptr inliers(new pcl::PointIndices());

            seg.setInputCloud(tmpCloud);
            seg.segment(*inliers, *coefficients);
            if(inliers->indices.size() < minInliersLine)
                continue;

            std::vector<int> nxtIndices;
            for (auto pt: inliers->indices)
            {
                nxtIndices.push_back(points2lines.at(iLine).at(pt)); 
            }
            points2lines.at(iLine) = nxtIndices;
            MyLine ln(*coefficients);
            Vec3 abcCoeffs = ln.getProjection(K_mat);
            Vec2 projNormal = Vec2(-abcCoeffs(1), abcCoeffs(0)).normalized();

            double angle2line2d = std::acos(projNormal.dot(endpoints2Normal(allLines.at(iLine))));

            if (CLIP_ANGLE(angle2line2d) > PARAMS::tDeltaAngle3d2d)
                continue;

            ++count;
            finalLines.push_back(iLine);

        }
    }

    pcl::SACSegmentation<pcl::PointXYZ> seg2;
    seg2.setOptimizeCoefficients(true);
    seg2.setModelType(pcl::SACMODEL_LINE);
    seg2.setMethodType(pcl::SAC_RANSAC);
    seg2.setDistanceThreshold(threshDistLineRansac);
    seg2.setMaxIterations(maxIterationsLineRansac);

    for (size_t i=0;i<visibility.size();++i)
        if (!visibility.at(i))
            cv::circle(img, cv::Point(edgesInFov.at(i).second(0), edgesInFov.at(i).second(1)), 1, cv::Scalar(175,175,175));

    std::cerr << "Before final merge there were : " <<  count << " lines" << std::endl;
    std::vector<bool> isSeen(sLine);

    for (unsigned int i=0;i< sLine; ++i)
        isSeen.at(i) = false;

    int ccId(0);
    PointCloudXYZRGB::Ptr endpointsDebug(new PointCloudXYZRGB);

    for (auto lineA : finalLines){
        if (!isSeen.at(lineA)){
            isSeen.at(lineA) = true;
            std::stack<int> q;
            q.push(lineA);
            std::vector<int> curCC;
            curCC.push_back(lineA);
            while(q.size()){
                int curElem = q.top();
                q.pop();

                for(auto lineB : linesAdjacency.at(curElem)){
                    if (!isSeen.at(lineB) && std::find(finalLines.begin(), finalLines.end(), lineB) != finalLines.end()
                        && angleBetweenLines(allLines.at(lineA), allLines.at(lineB)) < PARAMS::tMergeDeltaAngle){
                            // This is not enough (e.g we have 2 // lines but w/ different depths)
                            // That often appears when the lines are // so we can check at the distance between endpoints projected _|_
                            Vec2 dir = endpoints2Normal(allLines.at(lineA));
                            Vec2 norm = Vec2(-dir(1), dir(0));
                            Vec2 projP1, projP2;
                            projectPoint2Line2D(allLines.at(lineB).first, allLines.at(lineA).first, dir, projP1);
                            projectPoint2Line2D(allLines.at(lineB).second, allLines.at(lineA).first, dir, projP2);
                            double orthoDist = std::min(std::fabs((projP1-allLines.at(lineB).first).dot(norm)), std::fabs((projP2 - allLines.at(lineB).second).dot(norm)));

                            if (orthoDist < PARAMS::tOrthoDistLineMerge){
                                std::cerr << "We should merge " << lineA << " and " << lineB << std::endl;
                                q.push(lineB);
                                isSeen.at(lineB) = true;
                                curCC.push_back(lineB);
                            }
                        }
                    }
                }

            PointCloudPtr<pcl::PointXYZ> tmpCloud(new PointCloudXYZ);
            std::vector<int> allPointsCC;
            Vec2 refPoint = allLines.at(curCC.at(0)).first;
            for (auto lineCC : curCC){
                Vec2 normal = endpoints2Normal(allLines.at(lineCC));

               for (auto point: points2lines.at(lineCC)){
                   tmpCloud->push_back(pcl::PointXYZ(edgeCloud->points[edgesInFov.at(point).first].x, edgeCloud->points[edgesInFov.at(point).first].y, edgeCloud->points[edgesInFov.at(point).first].z));
                   allPointsCC.push_back(point);
               }
            }
            pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
            pcl::PointIndices::Ptr inliers(new pcl::PointIndices());

            seg2.setInputCloud(tmpCloud);
            seg2.segment(*inliers, *coefficients);
            MyLine ln(*coefficients);
            int count(0);

            double scoreMaxi(-1);
            double scoreMini(std::numeric_limits<double>::max());
            int iMini(-1);
            int iMaxi(-1);
            Vec3 abcCoeffs = ln.getProjection(K_mat);
            Vec2 projNormal = Vec2(-abcCoeffs(1), abcCoeffs(0)).normalized();

            cv::Scalar curColor(PARAMS::b[(ccId*colorMapMultiplier)%colorMapLength]*255,PARAMS::g[(ccId*colorMapMultiplier)%colorMapLength]*255,PARAMS::r[(ccId*colorMapMultiplier)%colorMapLength]*255);
            for (auto lineCC : curCC){
                // cv::line(img, cv::Point(allLines.at(lineCC).first(0), allLines.at(lineCC).first(1)), cv::Point(allLines.at(lineCC).second(0), allLines.at(lineCC).second(1)), curColor);
                double scoreS = getCoeffLine(allLines.at(lineCC).first, projNormal, refPoint);
                if (scoreS < scoreMini){
                    scoreMini = scoreS;
                    iMini = count;
                }
                if (scoreS > scoreMaxi){
                    scoreMaxi = scoreS;
                    iMaxi = count;
                }
                ++count;
                double scoreE = getCoeffLine(allLines.at(lineCC).second, projNormal, refPoint);
                if (scoreE < scoreMini){
                    scoreMini = scoreE;
                    iMini = count;
                }
                if (scoreE > scoreMaxi){
                    scoreMaxi = scoreE;
                    iMaxi = count;
                }
                ++count;
                std::cerr << scoreMini << " " << scoreMaxi << std::endl;
            }
            Vec2 endpointsA, endpointsB;
            std::cerr << iMini << " " << iMaxi << std::endl;
            if (iMini%2)
                endpointsA = allLines.at(curCC.at((int)(iMini/2))).second;
            else
                endpointsA = allLines.at(curCC.at((int)(iMini/2))).first;

            if (iMaxi%2)

                endpointsB = allLines.at(curCC.at((int)(iMaxi/2))).second;
            else
                endpointsB = allLines.at(curCC.at((int)(iMaxi/2))).first;


            if(inliers->indices.size() < minInliersLine)
                continue;

            ++ccId;
            MyLine curLine(*coefficients);
            //It would be helpful the 3D position of the endpoints. But we don't necessarily have depth info at that point
            // Since we know the equation of the 3D line, we look for the intersection point between the epipolar line and 3d line Maybe it's overkilled?
            Vec3 endpointACamera = getEndpointLocation(curLine, endpointsA, K);
            Vec3 endpointBCamera = getEndpointLocation(curLine, endpointsB, K);
            // ... and we project to the *world* frame
            const Mat3& cam_R = pose.rotation();
            const Vec3 cam_t = pose.translation();

            Mat34 projMatrix;
            openMVG::P_From_KRt(K, cam_R, cam_t, &projMatrix);
            Eigen::Matrix4d world2camera =  Eigen::Matrix4d::Identity();
            world2camera.block(0,0,3,3) = cam_R;
            world2camera.block(0,3,3,1) = cam_t;       
            
            Vec3 endpointsProjA = (world2camera.inverse() * endpointACamera.homogeneous()).block(0,0,3,1);
            Vec3 endpointsProjB = (world2camera.inverse() * endpointBCamera.homogeneous()).block(0,0,3,1);
            auto finalEndpoints2d =  std::make_pair(endpointsA, endpointsB);
            auto finalEndpoints3d =  std::make_pair(endpointsProjA, endpointsProjB); 
            std::cerr << finalEndpoints3d.first << std::endl;
            std::cerr << finalEndpoints3d.second << std::endl;
            /** Debug 
            Vec2 reprojEndpoints3d = projectW2I(finalEndpoints3d.first, projMatrix).hnormalized();
            Vec2 reprojEndpoints3d_b = projectW2I(finalEndpoints3d.second, projMatrix).hnormalized();
            std::cerr << reprojEndpoints3d << std::endl;
            std::cerr << reprojEndpoints3d_b << std::endl;

            cv::circle(img, cv::Point(reprojEndpoints3d(0), reprojEndpoints3d(1)), 2, curColor);
            cv::circle(img, cv::Point(reprojEndpoints3d_b(0), reprojEndpoints3d_b(1)), 2, curColor);
            **/
            if ((finalEndpoints3d.second - finalEndpoints3d.first).norm() > PARAMS::tMinLength3DSegment
                && (finalEndpoints3d.second - finalEndpoints3d.first).norm() < 10){
                std::vector<LBD::Descriptor> descs = allDesc.at(curCC.at(0)); // TODO: Update the descriptors!
                Segment3D curSegment(*coefficients, finalEndpoints2d, finalEndpoints3d, v->id_view, descs);
                result.push_back(curSegment);
                
                /** Debug 
                cv::line(img, cv::Point(endpointsA(0), endpointsA(1)), cv::Point(endpointsB(0), endpointsB(1)), curColor);
                for(auto& point: inliers->indices){
                    Vec2 curPt = edgesInFov.at(allPointsCC.at(point)).second.block(0,0,2,1);
                    cv::circle(img, cv::Point(curPt(0), curPt(1)), 1, curColor);
                    const pcl::PointXYZIRT& pclPt = edgeCloud->points[edgesInFov.at(allPointsCC.at(point)).first];
                    Vec3 ePt(pclPt.x, pclPt.y, pclPt.z);
                    Vec3 curPtInWF = (world2camera.inverse() * ePt.homogeneous()).block(0,0,3,1);
                    endpointsDebug->push_back(pcl::PointXYZRGB(curPtInWF.x(), curPtInWF.y(), curPtInWF.z(), curColor[2], curColor[1], curColor[0]));
                }
                endpointsDebug->push_back(pcl::PointXYZRGB(finalEndpoints3d.first.x(), finalEndpoints3d.first.y(), finalEndpoints3d.first.z(), curColor[2], curColor[1], curColor[0]));
                endpointsDebug->push_back(pcl::PointXYZRGB(finalEndpoints3d.second.x(), finalEndpoints3d.second.y(), finalEndpoints3d.second.z(), curColor[2], curColor[1], curColor[0]));
                **/
            }

        }
    }
    /** Debug
    std::cerr << sLine << " 2D segments were detected in the image and " << count << " will be used" << std::endl;
    cv::imshow("test",img);
    cv::waitKey(300);
    
    pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0, 0, 0);
    viewer->addPointCloud<pcl::PointXYZRGB> (endpointsDebug, "segments");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "segments");
    viewer->addCoordinateSystem (1.0);
    viewer->initCameraParameters ();

   while (!viewer->wasStopped ())
    {
      viewer->spinOnce (100);
    };
    **/ 
    
}

bool comparator ( const std::pair<double, std::pair<int, int>>& l, const std::pair<double, std::pair<int, int>>& r)
   { return l.first > r.first; }

std::vector<std::pair<double, std::pair<int, int>>> computeEdgeScoreOnLine(const cv::Mat& rangeMap, const bool& visualization){

 cv::Mat resultEdge(PARAMS::nRings, PARAMS::widthLidar, CV_32F, cv::Scalar(0));
 const float MAX_RANGE(1000.f);

 for(int i=0;i<PARAMS::nRings;++i){
     for(int j=PARAMS::nNeighborsSmoothness;j<PARAMS::widthLidar-PARAMS::nNeighborsSmoothness-1;++j){
         float sum = -(PARAMS::nNeighborsSmoothness*2+1)*rangeMap.at<float>(i,j);
          for(int k=-PARAMS::nNeighborsSmoothness;k<=PARAMS::nNeighborsSmoothness;++k){
              sum += rangeMap.at<float>(i,j+k);
          }
          if (rangeMap.at<float>(i,j) >= MAX_RANGE-0.001)  //Avoid infinite scores
             resultEdge.at<float>(i,j) = 0;
          else
             resultEdge.at<float>(i,j) = sum / (2*PARAMS::nNeighborsSmoothness*rangeMap.at<float>(i,j));
     }
 }
std::vector<std::pair<double, std::pair<int, int>>> edges;
std::vector<std::pair<double, std::pair<int, int>>> processedEdges;

 for (int i=0;i<PARAMS::nRings;++i)
     for(int j=0;j<PARAMS::widthLidar; ++j)
        if (resultEdge.at<float>(i,j) > PARAMS::tEdgeSmoothness)
            edges.push_back(std::make_pair(resultEdge.at<float>(i,j), std::make_pair(i,j)));

 //sort vector by desc smoothness score
 
 std::sort(edges.begin(), edges.end(), comparator);
 bool seen[PARAMS::nRings][PARAMS::widthLidar];
 for (auto i=0;i<PARAMS::nRings;++i)
    for (auto j=0;j<PARAMS::widthLidar;++j)
        seen[i][j] = false;
 // There's an issue with this part
 for (auto elem : edges){
     bool valid = true;
     for (int j = std::max(0,elem.second.second - PARAMS::nNeighborsSmoothness);j<std::min(PARAMS::widthLidar-1, elem.second.second + PARAMS::nNeighborsSmoothness);++j)
         if(seen[elem.second.first][j])
             valid = false;

     if (valid){
         //We add it to edges
         processedEdges.push_back(elem);
         seen[elem.second.first][elem.second.second] = true;
     }
 }
 std::cerr << " Found " << processedEdges.size() << " edges in current lidar scan" << std::endl;

 return processedEdges;
}

std::vector<int> projectPointCloud(const VelodynePointCloud::Ptr inputCloud, 
                                  const bool& visualization,
                                  const cv::Mat& img){

    const double angRes(360./PARAMS::widthLidar);
    const double MAX_RANGE(1000.f);
    cv::Mat rangeScan = cv::Mat(PARAMS::nRings, PARAMS::widthLidar, CV_32F, cv::Scalar::all(MAX_RANGE));

    double miniRange = std::numeric_limits<double>::max();
    double maxiRange = 0;
    std::map<std::pair<int, int>, int> colRowtoIdx;

    for(auto iPoint=0;iPoint < inputCloud->points.size(); ++iPoint)
    {
         auto it = &inputCloud->points[iPoint];
         int rowId = static_cast<int>(it->ring);
         pcl::PointXYZ curPt(it->x, it->y, it->z);
         Vec3 ePt(curPt.x, curPt.y, curPt.z);
         double range = ePt.norm();
         if (range < 1)
             continue;

         miniRange = std::min(miniRange, range);
         maxiRange = std::max(maxiRange, range);

         double horizontalAngle = std::atan2(curPt.x, curPt.y) * 180 / M_PI;
         int columnId = (int)(-std::round((horizontalAngle - 90.)/angRes) + PARAMS::widthLidar/2);

         if (columnId >= PARAMS::widthLidar)
             columnId -= PARAMS::widthLidar;
         columnId = PARAMS::widthLidar - columnId;
         rangeScan.at<float>(rowId, columnId) = float(range);
         colRowtoIdx[std::make_pair(rowId, columnId)] = iPoint;
    }

    std::vector<std::pair<double, std::pair<int, int>>> edgePoints = computeEdgeScoreOnLine(rangeScan, visualization);
    std::vector<int> result;
    
    pcl::PointCloud<pcl::PointXYZI>::Ptr edgePointCloud(new pcl::PointCloud<pcl::PointXYZI>);

    for(auto elem : edgePoints){
        result.push_back(colRowtoIdx[elem.second]);
        pcl::PointXYZIRT&  curIt(inputCloud->points[colRowtoIdx[elem.second]]);
        pcl::PointXYZI nIt(curIt.x, curIt.y, curIt.z, elem.first);
        edgePointCloud->push_back(nIt);
    }   
    // debugEdgeMap(img, edgePointCloud, d);
    return result;
}

void visualizeEndResult(PointCloudPtr<pcl::XPointXYZ> mergedCloud, 
                        const std::vector<std::vector<int>>& finalLines,
                        const std::vector<std::pair<int, Segment3D>>& allSegments,
                       std::map<int, int>& mapIdx)
{
    const int colMultiplier(7);
    PointCloudXYZRGB::Ptr segmentsEndpoints(new PointCloudXYZRGB);

    pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0, 0, 0);
    viewer->addPointCloud<pcl::XPointXYZ> (mergedCloud, "merged cloud");

    

    for (unsigned int i = 0 ; i < finalLines.size(); ++i)
    {
        Vec3 curColor(PARAMS::r[(colMultiplier * i)%255], PARAMS::g[(colMultiplier * i) % 255], PARAMS::b[(colMultiplier * i) % 255]);
        std::cerr << i << " " << finalLines.at(i).size() << std::endl;
        for (unsigned int j = 0 ; j < finalLines.at(i).size() ; ++j){
            const Segment3D& curSeg = allSegments.at(mapIdx[finalLines.at(i).at(j)]).second;

            segmentsEndpoints->push_back(pcl::PointXYZRGB(curSeg.endpoints3D.first.x(), curSeg.endpoints3D.first.y(), curSeg.endpoints3D.first.z(), 
            curColor.x()*255, curColor.y()*255, curColor.z()*255));

            segmentsEndpoints->push_back(pcl::PointXYZRGB(curSeg.endpoints3D.second.x(), curSeg.endpoints3D.second.y(), curSeg.endpoints3D.second.z(), 
            curColor.x()*255, curColor.y()*255, curColor.z()*255));

        }
    }

    viewer->addPointCloud<pcl::PointXYZRGB> (segmentsEndpoints, "segments");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "segments");
    viewer->addCoordinateSystem (1.0);
    viewer->initCameraParameters ();

   while (!viewer->wasStopped ())
    {
      viewer->spinOnce (100);
    };
}    

} // namespace sfm
} // namespace openMVG