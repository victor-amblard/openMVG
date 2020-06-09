#include "sfm_geometric_features_extraction.hpp"

#ifdef OPENMVG_USE_OPENMP
#include <omp.h>
#endif

namespace cv {
    // calculates the median value of a single channel
    // based on https://github.com/arnaudgelas/OpenCVExamples/blob/master/cvMat/Statistics/Median/Median.cpp
    double median(const cv::Mat& channel )
    {
        double m = (channel.rows*channel.cols) / 2;
        int bin = 0;
        double med = -1.0;

        int histSize = 256;
        float range[] = { 0, 256 };
        const float* histRange = { range };
        bool uniform = true;
        bool accumulate = false;
        cv::Mat hist;
        cv::calcHist( &channel, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, uniform, accumulate );

        for ( int i = 0; i < histSize && med < 0.0; ++i )
        {
            bin += cvRound( hist.at< float >( i ) );
            if ( bin > m && med < 0.0 )
                med = i;
        }

        return med;
    }
}

namespace openMVG {
namespace sfm {

void extractPlanesFromCloud(PointCloudXYZ::Ptr filteredCloud,
                            std::vector<pcl::ModelCoefficients>& planes,
                            std::vector<PointCloudXYZ::Ptr, Eigen::aligned_allocator<PointCloudXYZ::Ptr> >& outputClouds)
{

    //Parameters
    const int maxPlanes(10);
    const int minCloudSize(50);
    const float distanceThreshold(0.5);
    const int maxIterations(300);
    const Eigen::Vector3f axis(0,0,1.f);

    PointCloudXYZ::Ptr cloud_p(new PointCloudXYZ), cloud_f(new PointCloudXYZ);

   // Create the filtering object: downsample the dataset using a leaf size of 1cm

   pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
   pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
   pcl::SACSegmentation<pcl::PointXYZ> seg;
   seg.setOptimizeCoefficients(true);
   seg.setModelType(pcl::SACMODEL_PARALLEL_PLANE);
   seg.setMethodType(pcl::SAC_RANSAC);
   seg.setDistanceThreshold(distanceThreshold);
   seg.setMaxIterations(maxIterations);

   // Create the filtering object
   pcl::ExtractIndices<pcl::PointXYZ> extract;
   std::vector<PointCloudXYZ, Eigen::aligned_allocator<PointCloudXYZ> > clouds_vector;
   std::vector<pcl::ModelCoefficients> normalCoeff;

   seg.setAxis(axis);
   seg.setEpsAngle(30.0f * (M_PI/180.0f) );

   int i = 0, nr_points = (int)filteredCloud->points.size();

    while((i < maxPlanes || maxPlanes==0) && (cloud_p->points.size() > minCloudSize || i==0)){

       seg.setInputCloud(filteredCloud);
       pcl::ScopeTime scopeTime("Test loop");
       {
           seg.segment(*inliers, *coefficients);
       }
       if (inliers->indices.size() == 0)
       {
           std::cerr << "Could not estimate a planar model for the given dataset." << std::endl;
           break;
       }


       // Extract the inliers
       extract.setInputCloud(filteredCloud);
       extract.setIndices(inliers);
       extract.setNegative(false);
       extract.filter(*cloud_p);

       std::cerr << "PointCloud representing the planar component: " << cloud_p->width * cloud_p->height << " data points." << std::endl;

       extract.setInputCloud(filteredCloud);
       extract.setIndices(inliers);
       extract.setNegative(true);
       extract.filter(*filteredCloud);

       clouds_vector.push_back(*cloud_p);
       normalCoeff.push_back(*coefficients);

       ++i;
    }
    std::vector<Eigen::Vector6f> linesCoeff;
    findLinesFromPlanes(normalCoeff, linesCoeff);
    planes = normalCoeff;

    for (size_t j=0;j<clouds_vector.size();++j){
        PointCloudXYZ::Ptr curP(new PointCloudXYZ);
        *curP = clouds_vector.at(j);
        outputClouds.push_back(curP);
    }
}
void findLinesFromPlanes(const std::vector<pcl::ModelCoefficients>& planesCoeff,
                         std::vector<Eigen::Vector6f>& linesCoeff)
{
    for (size_t i=0;i<planesCoeff.size();++i){
        auto coeffI = planesCoeff.at(i);
        Eigen::Vector4f planeA(coeffI.values[0], coeffI.values[1], coeffI.values[2], coeffI.values[3]);

        for (size_t j=i+1;j<planesCoeff.size();++j){
            auto coeffJ = planesCoeff.at(j);
            Eigen::Vector4f planeB(coeffJ.values[0], coeffJ.values[1], coeffJ.values[2], coeffJ.values[3]);
            Eigen::VectorXf lineIJ;
            bool foundInter = pcl::planeWithPlaneIntersection(planeA, planeB, lineIJ);
            lineIJ.resize(6);
            linesCoeff.push_back(lineIJ);
        }
    }
}

float computeAngularGap(PointCloudXYZ::Ptr inlierCloud, 
                        const pcl::ModelCoefficients& planeCoeff, 
                        const pcl::PointXYZ& poi)
{
    // Find 2 orthonormal vectors u,v
    float curMaxi(0);
    float prevTheta(0);
    float curTheta(0);
    Eigen::Vector3f ePOI(poi.getVector3fMap());

    Eigen::Vector3f n(planeCoeff.values[0], planeCoeff.values[1], planeCoeff.values[2]);
    n.normalize();
    Eigen::Vector3f u(n(2), n(2), -n(0)-n(1));
    Eigen::Vector3f v = n.cross(u);
    u.normalize();
    v.normalize();

    // Compute max theta_{i+1}-theta_i
    for(auto it=inlierCloud->begin();it!=inlierCloud->end();++it){
        float ui = ((*it).getVector3fMap() - ePOI).dot(u);
        float vi = ((*it).getVector3fMap() - ePOI).dot(v);
        curTheta = std::atan(ui/vi);
        curMaxi = std::max(curMaxi, curTheta - prevTheta);
        prevTheta = curTheta;
    }

    return curMaxi;

}

void checkInlierLines(const std::vector<Eigen::Vector6f>& allLines,
                      PointCloudXYZ::Ptr pCloud,
                      std::vector<std::pair<Line, std::vector<int>>> * results)
{
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud (pCloud);
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setNegative(false);



    const int maxIterationsPlaneRansac(100);
    const float threshDist(3.f); //distance from plane/plane intersection
    const float angularGapThresh(M_PI/2);
    const int KNN(20); //# of NN considered in the local plane fitting -- depends on LIDAR resolution

    //RANSAC initialization for plane estimation
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.2);
    seg.setMaxIterations(maxIterationsPlaneRansac);

    std::vector<std::pair<int, int>> pointsOnLine;

    for (auto iCloud=0;iCloud < pCloud->points.size();++iCloud){

        pcl::PointXYZ curPoint(pCloud->points[iCloud]);
        #ifdef OPENMVG_USE_OPENMP
        #pragma omp parallel for
        #endif
        for (size_t iLine = 0;iLine < allLines.size(); ++iLine){

            Eigen::Vector3f lPoint = allLines.at(iLine).block(0,0,3,1);
            Eigen::Vector3f lDir = allLines.at(iLine).block(3,0,3,1);

            if (getDistToLine(lPoint, lDir, curPoint.getVector3fMap()) < threshDist){
                std::vector<int> pointIdxNKNSearch(KNN);
                std::vector<float> pointNKNSquaredDistance(KNN);
                if (kdtree.nearestKSearch(curPoint, KNN, pointIdxNKNSearch, pointNKNSquaredDistance) > 0){

                    PointCloudXYZ::Ptr tmpC(new PointCloudXYZ);
                    tmpC->push_back(curPoint);
                    for (auto iR = 0;iR < pointIdxNKNSearch.size();++iR){
                        tmpC->push_back(pCloud->points.at(pointIdxNKNSearch.at(iR)));
                    }

                    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
                    pcl::PointIndices::Ptr inliers(new pcl::PointIndices());

                    seg.setInputCloud(tmpC);
                    seg.segment(*inliers, *coefficients);
                    PointCloudXYZ::Ptr inliersCloud(new PointCloudXYZ);

                    if (inliers->indices.size() > 0){
                        extract.setInputCloud(tmpC);
                        extract.setIndices(inliers);
                        extract.filter(*inliersCloud);
                        bool valid = std::find(inliers->indices.begin(), inliers->indices.end(), 0) != inliers->indices.end();
                        if (valid){
                            if (computeAngularGap(inliersCloud, *coefficients, curPoint) > angularGapThresh){
                                #ifdef OPENMVG_USE_OPENMP
                                #pragma omp critical
                                #endif
                                pointsOnLine.push_back(std::make_pair(iCloud, iLine));
                            }
                        }
                    }
                }
            }
        }
    }

    checkValidPointsOnLines(allLines, pCloud, pointsOnLine, results);
}

void checkValidPointsOnLines(const std::vector<Eigen::Matrix<float,6,1>>& allLines,
                            PointCloudXYZ::Ptr pCloud,
                            std::vector<std::pair<int, int>>& pointsOnLine,
                            std::vector<std::pair<Line, std::vector<int>>> * results)
{
    const float threshDistLineRansac(0.2);
    const int maxIterationsLineRansac(100);
    const int minInliersLine(10);

    //Initialization line RANSAC
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_LINE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(threshDistLineRansac);
    seg.setMaxIterations(maxIterationsLineRansac);

    std::cout << " Before removal there are " << allLines.size() << " lines" << std::endl;
    for (auto iL=0;iL<allLines.size();++iL){

        PointCloudXYZ::Ptr tmpCloud(new PointCloudXYZ);
        std::vector <int> idxTmp;

        for (auto i=0;i<pointsOnLine.size();++i){
            if (pointsOnLine.at(i).second == iL){
                tmpCloud->push_back(pCloud->points[pointsOnLine.at(i).first]);
                idxTmp.push_back(pointsOnLine.at(i).first);
            }
         }
        if (!tmpCloud->points.size())
            continue;

        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices());

        seg.setInputCloud(tmpCloud);
        seg.segment(*inliers, *coefficients);
        if(inliers->indices.size() < minInliersLine)
            continue;

        std::vector<int> validIdxLine;

        for (auto jIndices=0;jIndices<inliers->indices.size();++jIndices){
            validIdxLine.push_back(idxTmp.at(inliers->indices[jIndices]));
        }

        Line ln(*coefficients);
        results->push_back(std::make_pair(ln, validIdxLine));
    }
}

void getLinesInImageAfm(const std::string& filename,
                        std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>>& allLines,
                        const bool& visualization)
{
    const float minLineLength(10.f);

    std::ifstream inFile;
    inFile.open(filename);

    if (!inFile)
        std::cerr << "Unable to open " << filename << std::endl;

    std::string line;
    std::getline(inFile, line); //ignore first line = #lines

    while (std::getline(inFile, line)){
        std::stringstream ss(line);
        double value;
        double line[4] = {0.,0.,0.,0.};
        int iCount(0);
        while (ss >> value){
            line[iCount] = value;
            ++iCount;
        }
        if (iCount!=4)
            std::cerr << "Error when reading line from txt file!" << std::endl;
        else
            if (pow(line[2]-line[0],2.f)+pow(line[3]-line[1], 2.f) > pow(minLineLength,2.f))
                allLines.push_back(std::make_pair(Eigen::Vector2d(line[0], line[1]), Eigen::Vector2d(line[2], line[3])));
    }
}

void getLinesInImageCanny(const std::string& imgFn,
                          std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>>& allLines,
                          const bool& visualization)
{
    const cv::Mat img(cv::imread(imgFn,0));

    cv::Mat dst, cdst;

    double v = cv::median(img);
    int lower = int(std::max(0., (1.0 - 0.6) * v));
    int upper = int(std::min(255., (1.0 + 0.6) * v));

    cv::Canny(img, dst, lower,upper,3);
    std::vector<cv::Vec4i> lines;

    cv::HoughLinesP(dst, lines, 1, 2*M_PI/180, 50, 60, 25);

    cv::cvtColor(img, cdst, CV_GRAY2BGR);

    for( size_t i = 0; i < lines.size(); i++ )
    {
        cv::Point pt1, pt2;

        pt1.x = std::lrint(lines[i][0]);
        pt1.y = std::lrint(lines[i][1]);
        pt2.x = std::lrint(lines[i][2]);
        pt2.y = std::lrint(lines[i][3]);
        if (visualization)
            cv::line(cdst, pt1, pt2, cv::Scalar(0,0,255), 1, CV_AA);

      allLines.push_back(std::make_pair(Eigen::Vector2d(pt1.x, pt1.y), Eigen::Vector2d(pt2.x, pt2.y)));
    }
    if (visualization){
        cv::imshow("canny", dst);
        cv::waitKey();
        cv::imshow("detected lines", cdst);
        cv::waitKey(0);
   }

}
void loadAllAfMFiles(const std::vector<std::string>& filenames,
                     std::vector<std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>>>& allLines)
{
    for (size_t i=0;i<filenames.size();++i){
        const std::string& filename(filenames.at(i));
        getLinesInImageAfm(filename, allLines.at(i));
    }
}
std::pair<int,float> matchLine2Line(const std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>>& detectedLines,
                                    const Eigen::Vector3d backProjectedLine)
{
    const double thetaThreshold(25*M_PI/180); 
    const double distThreshold(100); 
    return std::make_pair(0,0.f);
    /*
    Eigen::Vector2f curLDir(1, (proj3DLine.second(1)-proj3DLine.first(1))/(proj3DLine.second(0)-proj3DLine.first(0)));
    Eigen::Vector2f tentLDir(1, (detected2DLine.second(1)-detected2DLine.first(1))/(detected2DLine.second(0)-detected2DLine.first(0)));
    curLDir.normalize();
    tentLDir.normalize();

    double thetaDist = std::acos(curLDir.dot(tentLDir)/(curLDir.norm()*tentLDir.norm()));
    double dDist = pointToLineDist(detected2DLine.first, proj3DLine) + pointToLineDist(detected2DLine.second, proj3DLine);
    //TODO: Just normalize it once
    double tEnd = getParameterPointLine2(proj3DLine.first, proj3DLine.second, curLDir);

    Eigen::Vector2f startProj, endProj;
    //Projection of 2D line endpoints to backprojected 3D line

    projectPoint2Line2D(detected2DLine.first, proj3DLine.first, curLDir, startProj);
    projectPoint2Line2D(detected2DLine.second, proj3DLine.first, curLDir, endProj);


    double tStartProj = getParameterPointLine2(proj3DLine.first, startProj, curLDir);
    double tEndProj = getParameterPointLine2(proj3DLine.first, endProj, curLDir);
    if (tEndProj < tStartProj){
        double tmp = tStartProj;
        tStartProj = tEndProj;
        tEndProj = tmp;
    }

    double alphaEnd = std::max(0., std::min(1., tEndProj/tEnd));
    double alphaStart = std::max(0., std::min(1., tStartProj/tEnd));

    double lenOverlap = alphaEnd-alphaStart;

    bool overlaps = (lenOverlap > 0.2); //20% overlap at least

    if (thetaDist < thetaThreshold && dDist < distThreshold && overlaps){
        distance(0) = thetaDist;
        distance(1) = dDist;
        distance(2) = lenOverlap;
        return std::make_pair(true, distance.norm());
    } else {
        return std::make_pair(false;

    }*/
}

void visualize3dLines(const std::vector<std::pair<Line, std::vector<int>>>* lines,
                      PointCloudXYZ::Ptr& cloud,
                      std::vector<Eigen::Vector6f>& intersections)
{
    pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0, 0, 0);
    pcl::RGB rgb_color;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr coloredCloud(new pcl::PointCloud<pcl::PointXYZRGB>);

    int k(0);

    for (size_t i=0;i<lines->size();++i){
        auto elem = lines->at(i);
        std::vector<int> curIdx = elem.second;
        rgb_color = pcl::GlasbeyLUT::at(k);
        for (auto pt: curIdx){
            pcl::PointXYZ curPoint(cloud->points[pt]);
            coloredCloud->push_back(pcl::PointXYZRGB(curPoint.x, curPoint.y, curPoint.z, rgb_color.r, rgb_color.g, rgb_color.b));
        }
        ++k;
    }

    viewer->addPointCloud<pcl::PointXYZ>(cloud, "global");
    viewer->addPointCloud<pcl::PointXYZRGB>(coloredCloud, "lines");

    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "lines");
    while (!viewer->wasStopped ())
     {
       viewer->spinOnce (100);
     };
}


bool isLineOccluded(void)
{
    //TODO
    return false;
}
int getLineLineCorrespondence(const Eigen::Vector4d& cur2dSegment,
                              const std::vector<std::pair<IndexT, Eigen::Vector4d>> allVisible3dlines,
                              const Mat3& K,
                              const geometry::Pose3& pose)
{
    
    for(const auto&elem: allVisible3dlines){
        IndexT idx3dline = elem.first;
        Eigen::Vector4d endpoints_3d = elem.second;

    }
    return 0;
}
bool isLineInFOV(const Eigen::Vector6d& line,
                 const IndexT width, 
                 const IndexT height,
                 const Mat3& K,
                 const geometry::Pose3& pose,
                 Eigen::Vector4d& endPoints)
{
   
    //TODO: How to ensure that the line is not behind the camera?

    Mat34 projMatrix;
    openMVG::P_From_KRt(K, pose.rotation(), pose.translation(), &projMatrix);
    const Vec3& ep_s(line.block(0,0,3,1));
    const Vec3& ep_e(line.block(3,0,3,1));


    Eigen::Vector2d projEndA = openMVG::Project(projMatrix, ep_s);
    Eigen::Vector2d projEndB = openMVG::Project(projMatrix, ep_e);
    double coefDir = (projEndB(1)-projEndA(1))/(projEndB(0)-projEndA(0));

    const Eigen::Vector3d lineCoeffs = Eigen::Vector3d(coefDir, 1, -coefDir*projEndA(0)-projEndA(1));
    Eigen::Vector2d p1;
    Eigen::Vector2d p2;
    p1(1) = 0;
    p2(1) = height;
    p1(0) = (-lineCoeffs(1)*p1.y()-lineCoeffs(2))/lineCoeffs(0);
    p2(0) = (-lineCoeffs(1)*p2.y()-lineCoeffs(2))/lineCoeffs(0);

    if (projEndA(1) > projEndB(1)){
        Eigen::Vector2d tmp;
        tmp = projEndB;
        projEndB = projEndA;
        projEndA = tmp;
    }
    bool ok = true;
    if (projEndA(0) > width || projEndA(1) < 0 || projEndA(1) > height || projEndA(0) <0){
        endPoints.block(0,0,2,1) = p1;
        ok=false;
    }else
        endPoints.block(0,0,2,1) = projEndA.block(0,0,2,1);

    if (projEndB(0) > width || projEndB(1) < 0 || projEndB(1) > height || projEndB(0) <0 )
        if (!ok)
            return false;
        else
            endPoints.block(2,0,2,1) = p2;
    else
        endPoints.block(2,0,2,1) = projEndB.block(0,0,2,1);


    if (((endPoints(0)>=0 && endPoints(0) < width) || (endPoints(2)>=0 && endPoints(2) < width)))
        return true;

    return false;
}
std::vector<std::pair<IndexT, Eigen::Vector4d>> getAllVisibleLines(const Hash_Map<IndexT, Eigen::Vector6d>& all_3d_lines,
                        const geometry::Pose3& pose,
                        const Mat3& intr,
                        const View * view)
{
    std::vector<std::pair<IndexT, Eigen::Vector4d>> visibleLines;

    for (auto& elem: all_3d_lines){ 
        const auto curIdx = elem.first;
        const Eigen::Vector6d& l = elem.second;
        Eigen::Vector4d lineEndpoints;
        std::cerr << view->id_view << std::endl;
        if(isLineInFOV(l, view->ui_width, view->ui_height, intr, pose, lineEndpoints)){
            if (!isLineOccluded()){
                std::cout << "Visible!" << std::endl;
                visibleLines.push_back(std::make_pair(curIdx, lineEndpoints));
            }
        }
    }
    return visibleLines;
}
void visualizeMatches(const std::vector<std::pair<IndexT, std::vector<IndexT>>>& matches,
                      const Hash_Map<IndexT, Eigen::Vector6d>& all_3d_lines,
                      const Hash_Map<IndexT, std::pair<IndexT, Eigen::Vector4d>>& all_2d_lines,
                      const Hash_Map<IndexT, Hash_Map<IndexT, Eigen::Vector4d>>& proj_3d_lines,
                      const View * v,
                      const std::string& rootPath)
{
    IndexT viewId = v->id_view;
    std::string fn = rootPath + "/"+v->s_Img_path;
    cv::Mat img = cv::imread(fn);

    for(size_t i=0;i<matches.size();++i){
        bool valid3dLine = false;
        IndexT line3dIdx = matches.at(i).first;
        for (IndexT line2dIdx: matches.at(i).second){
            if (all_2d_lines.at(line2dIdx).first == viewId){
                valid3dLine = true;
                Eigen::Vector4d line_2d_endpoints = all_2d_lines.at(line2dIdx).second;
                cv::line(img, cv::Point(line_2d_endpoints(0), line_2d_endpoints(1)), cv::Point(line_2d_endpoints(2), line_2d_endpoints(3)), cv::Scalar(0,0,255), 1, CV_AA);
            }
        }
        if (valid3dLine){
            const Eigen::Vector4d line_3d_endpoints = proj_3d_lines.at(line3dIdx).at(viewId);
            cv::line(img, cv::Point(line_3d_endpoints(0), line_3d_endpoints(1)), cv::Point(line_3d_endpoints(2), line_3d_endpoints(3)), cv::Scalar(255,0,0), 1, CV_AA);

        }
    }

    cv::imshow(v->s_Img_path, img);
    cv::waitKey(0);
}
}
}