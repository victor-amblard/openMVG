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
double projectPointOnLine2(const Eigen::Vector2d& point,
                         const Eigen::Vector2d& pointOnLine,
                         const Eigen::Vector2d& lineDirNormd,
                         Eigen::Vector2d& targetPoint)
{
    Eigen::Vector2d displacement(point-pointOnLine);
    double coef = displacement.dot(lineDirNormd);
    targetPoint = pointOnLine + coef*lineDirNormd;
    return coef;
}
double matchLine2Line(const Eigen::Vector4d& segment_2d,
                      const Eigen::Vector4d& projected_3d_line)
{
    const double thetaThreshold(30*M_PI/180); //Very conservative thresholds 
    const double distThreshold(100); 
    const double overlapThreshold(0.3);
    const double NO_MATCH(-1);

    //Just as a convention, we set the starting point to be the lower x point
    Eigen::Vector2d startPoint2d, endPoint2d, startPoint3d, endPoint3d;
    startPoint2d = segment_2d.block(0,0,2,1);
    endPoint2d = segment_2d.block(2,0,2,1);
    startPoint3d = projected_3d_line.block(0,0,2,1);
    endPoint3d = projected_3d_line.block(2,0,2,1);

    if (segment_2d(0) > segment_2d(2)){
        startPoint2d = segment_2d.block(2,0,2,1);
        endPoint2d = segment_2d.block(0,0,2,1);
    }
    if (projected_3d_line(0) > projected_3d_line(2)){
        startPoint3d = projected_3d_line.block(2,0,2,1);
        endPoint3d = projected_3d_line.block(0,0,2,1);
    }

    Eigen::Vector2d normalizedSegmentDir = (endPoint2d - startPoint2d).normalized();
    Eigen::Vector2d normalizedLineDir = (endPoint3d - startPoint3d);
    double tMax = normalizedLineDir.norm();
    normalizedLineDir.normalize();
    Eigen::Vector3d abcLine;

    if (endPoint3d(0)-startPoint3d(0) > std::numeric_limits<double>::min()){
        double alpha = (endPoint3d(1)-startPoint3d(1))/(endPoint3d(0)-startPoint3d(0));
        abcLine = Eigen::Vector3d(alpha,-1,startPoint3d(1)-alpha*startPoint3d(0));
    }else{
        abcLine = Eigen::Vector3d(1,0,-startPoint3d(0));
    }

    // Step 1: Estimate 2d endpoints' projection on projected 3d line
    double tStart, tEnd;
    Eigen::Vector2d startProj, endProj;
    tStart = std::max(0., std::min(tMax, projectPointOnLine2(startPoint2d, startPoint3d, normalizedLineDir, startProj)));
    tEnd = std::max(0., std::min(tMax, projectPointOnLine2(endPoint2d, startPoint3d, normalizedLineDir, endProj)));
    // Step 2: Deduce the overlap length
    double lenOverlap = std::fabs(tEnd - tStart)/tMax;

    // Step 3: Get the direction difference
    double theta = std::acos(normalizedSegmentDir.dot(normalizedLineDir));

    // Step 4: Get the distance
    Eigen::Vector3d middlePoint = (1/2*(startPoint2d + endPoint2d)).homogeneous();
    double dist = std::fabs(abcLine.dot(middlePoint))/(abcLine.block(0,0,2,1)).norm();

    if (theta < thetaThreshold && dist < distThreshold && lenOverlap > overlapThreshold){
        return Eigen::Vector3d(theta/thetaThreshold, dist/distThreshold, lenOverlap/overlapThreshold).norm(); 
    }else{
        std::cerr << Eigen::Vector3d(theta, dist, lenOverlap) << std::endl;
        return NO_MATCH;
    }
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
                              const Hash_Map<IndexT, Eigen::Vector4d>  allVisible3dlines,
                              const Mat3& K,
                              const geometry::Pose3& pose,
                              IndexT id)
{
    int iMini(-1);
    double miniScore(std::numeric_limits<double>::max());

    for(const auto&elem: allVisible3dlines){
        IndexT idx3dline = elem.first;
        Eigen::Vector4d endpoints_3d = elem.second;
        if (double curScore = matchLine2Line(cur2dSegment, endpoints_3d) >= 0){
            if (curScore < miniScore){
                miniScore = curScore;
                iMini = idx3dline;
            }
        } 
    }
    return iMini;
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
Hash_Map<IndexT, Eigen::Vector4d> getAllVisibleLines(const Hash_Map<IndexT, Eigen::Vector6d>& all_3d_lines,
                        const geometry::Pose3& pose,
                        const Mat3& intr,
                        const View * view)
{
    Hash_Map<IndexT, Eigen::Vector4d> visibleLines;

    for (auto& elem: all_3d_lines){ 
        const auto curIdx = elem.first;
        const Eigen::Vector6d& l = elem.second;
        Eigen::Vector4d lineEndpoints;
        std::cerr << view->id_view << std::endl;
        if(isLineInFOV(l, view->ui_width, view->ui_height, intr, pose, lineEndpoints)){
            if (!isLineOccluded()){
                std::cout << "Visible!" << std::endl;
                visibleLines[curIdx] =  lineEndpoints;
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
                cv::line(img, cv::Point(line_2d_endpoints(0), line_2d_endpoints(1)), cv::Point(line_2d_endpoints(2), line_2d_endpoints(3)), cv::Scalar(0,0,255), 2, CV_AA);
            }
        }
        if (valid3dLine){
            const Eigen::Vector4d line_3d_endpoints = proj_3d_lines.at(line3dIdx).at(viewId);
            cv::line(img, cv::Point(line_3d_endpoints(0), line_3d_endpoints(1)), cv::Point(line_3d_endpoints(2), line_3d_endpoints(3)), cv::Scalar(255,0,0), 2, CV_AA);

        }
    }

    cv::imshow(v->s_Img_path, img);
    cv::waitKey(0);
}
void testLineReprojectionCostFunction(const double * const cam_intrinsics,
                                      const double * const cam_extrinsics,
                                      const double * const line_3d_endpoint,
                                      const double * m_line_2d_endpoints,
                                      const View * v)
{
    const double * cam_R = cam_extrinsics;
    Eigen::Map<const Eigen::Matrix<double, 3, 1>> cam_t(&cam_extrinsics[3]);

    Eigen::Matrix<double, 3, 1> transformed_point_start, transformed_point_end;
    // Rotate the point according the camera rotation
    
    ceres::AngleAxisRotatePoint(cam_R, line_3d_endpoint, transformed_point_start.data()); //3D starting point
    ceres::AngleAxisRotatePoint(cam_R, &line_3d_endpoint[3], transformed_point_end.data()); //3D end point

    // Apply the camera translation
    transformed_point_start += cam_t;
    transformed_point_end += cam_t;

    // Transform the point from homogeneous to euclidean (undistorted point)
    Eigen::Matrix<double, 2, 1> projected_point3d_start = transformed_point_start.hnormalized();
    Eigen::Matrix<double, 2, 1> projected_point3d_end = transformed_point_end.hnormalized();

    //--
    // Apply intrinsic parameters
    //--

    const double& focal = cam_intrinsics[0];
    const double& principal_point_x = cam_intrinsics[1];
    const double& principal_point_y = cam_intrinsics[2];
    // Apply focal length and principal point to get the final image coordinates
    Eigen::Matrix<double,2,1> proj_3d_point_start(principal_point_x + projected_point3d_start.x() * focal,
                                            principal_point_y + projected_point3d_start.y() * focal);

    Eigen::Matrix<double,2,1> proj_3d_point_end(principal_point_x + projected_point3d_end.x() * focal,
                                           principal_point_y + projected_point3d_end.y() * focal);

    //Compute orthogonal projection of 2D point on a 2D line
    const Eigen::Matrix<double,2,1> line_2d_start(m_line_2d_endpoints[0], m_line_2d_endpoints[1]);
    const Eigen::Matrix<double,2,1> line_2d_end(m_line_2d_endpoints[2], m_line_2d_endpoints[3]);
    
    
    //We project on m_line_2d_endpoint_start and end on the (finite) line formed by (proj_3d_point_start, proj_3d_point_end)
    Eigen::Matrix<double,2,1> start_2d_proj_2d_line ;
    Eigen::Matrix<double,2,1> end_2d_proj_2d_line;

    std::cerr << proj_3d_point_start << std::endl;
    std::cerr << proj_3d_point_end << std::endl;
    std::cerr << " **** " << std::endl;
    //Step 1: Compute the projection 
    Eigen::Matrix<double,2,1> proj_3d_dir = proj_3d_point_end - proj_3d_point_start;


    double t_start = (line_2d_start-proj_3d_point_start).y()/proj_3d_dir.y(); 
    double t_end = (line_2d_end-proj_3d_point_start).y()/proj_3d_dir.y();

    t_start = std::min(double(1), std::max(double(0),t_start));
    t_end = std::min(double(1), std::max(double(0), t_end));

    std::cerr << t_start << " " << t_end << " " << std::endl;

    start_2d_proj_2d_line = proj_3d_point_start + t_start*proj_3d_dir;
    end_2d_proj_2d_line = proj_3d_point_start + t_end*proj_3d_dir;

    Eigen::Matrix<double,2,1> dir_2d = (line_2d_end - line_2d_start).normalized();

    Eigen::Matrix<double,2,1> adjusted_2d_start = line_2d_start+(start_2d_proj_2d_line-line_2d_start).dot(dir_2d)*dir_2d;
    Eigen::Matrix<double,2,1> adjusted_2d_end = line_2d_start+(end_2d_proj_2d_line-line_2d_start).dot(dir_2d)*dir_2d;

    cv::Mat test = cv::imread("/home/victor/Data/Stages/MIT/newer_college_dataset/subset/infra1/subset/"+v->s_Img_path);
    cv::line(test, cv::Point(m_line_2d_endpoints[0], m_line_2d_endpoints[1]), cv::Point(m_line_2d_endpoints[2], m_line_2d_endpoints[3]), cv::Scalar(255,0,0), 1, CV_AA);
    cv::line(test, cv::Point(proj_3d_point_start.x(), proj_3d_point_start.y()), cv::Point(proj_3d_point_end.x(), proj_3d_point_end.y()), cv::Scalar(255,0,0), 1, CV_AA);
    
    // cv::circle(test, cv::Point(proj_3d_point_start.x(), proj_3d_point_start.y()), 3,cv::Scalar(0,255,0),2);
    // cv::circle(test, cv::Point(proj_3d_point_end.x(), proj_3d_point_end.y()), 3,cv::Scalar(0,0,255),2);

    cv::circle(test, cv::Point(start_2d_proj_2d_line.x(), start_2d_proj_2d_line.y()), 3,cv::Scalar(0,255,0),2);
    cv::circle(test, cv::Point(end_2d_proj_2d_line.x(), end_2d_proj_2d_line.y()), 3,cv::Scalar(0,0,255),2);

    cv::circle(test, cv::Point(adjusted_2d_start.x(), adjusted_2d_start.y()), 3,cv::Scalar(0,255,0),2);
    cv::circle(test, cv::Point(adjusted_2d_end.x(), adjusted_2d_end.y()), 3,cv::Scalar(0,0,255),2);


    cv::imshow("test",test);
    cv::waitKey(0);
    std::cerr << "Cost start: " << pow(start_2d_proj_2d_line.x() - adjusted_2d_start.x(),2)+ (start_2d_proj_2d_line.y() - adjusted_2d_start.y(),2)<< std::endl;
    std::cerr << "Cost end: " << pow(end_2d_proj_2d_line.x() - adjusted_2d_end.x(),2)+ pow(end_2d_proj_2d_line.y() - adjusted_2d_end.y(),2) << std::endl;

}
void testLineReprojectionPlucker(const double * const cam_intrinsics,
                                      const double * const cam_extrinsics,
                                      const double * const line_3d_endpoint,
                                      const double * m_line_2d_endpoints,
                                      const View * v)
{
    const double * cam_R = cam_extrinsics;
    Mat3 mat_r;
    ceres::AngleAxisToRotationMatrix(cam_R, mat_r.data());

    Eigen::Map<const Eigen::Matrix<double, 3, 1>> cam_t(&cam_extrinsics[3]);

    Eigen::Matrix<double,3,3> K;
    K << double(cam_intrinsics[0]), double(0), double(cam_intrinsics[1]),
         double(0), double(cam_intrinsics[0]), double(cam_intrinsics[2]),
         double(0), double(0), double(1);

    Eigen::Matrix<double,4,4> projMat = Eigen::Matrix<double,4,4>::Zero();
    Eigen::Matrix<double,4,4> RT_mat =  Eigen::Matrix<double,4,4>::Identity();
    RT_mat.block(0,0,3,3) = mat_r;
    RT_mat.block(0,3,3,1) = cam_t;
    std::cerr << RT_mat << std::endl;
    const Eigen::Matrix<double,3,4> Pmat = K * RT_mat.block(0,0,3,4);
    projMat.block(0,0,3,4) = Pmat;
    projMat(3,3) = double(1);

    Eigen::Matrix<double,4,4>pluckerMatrix = Eigen::Matrix<double,4,4>::Zero();

    for(size_t i=1;i<4;++i){
      pluckerMatrix(i,0) = line_3d_endpoint[i-1];
    }
    pluckerMatrix(2,1) = line_3d_endpoint[3];
    pluckerMatrix(3,1) = line_3d_endpoint[4];
    pluckerMatrix(3,2) = line_3d_endpoint[5];

    for(size_t i=0;i<4;++i)
        for(size_t j=i;j<4;++j)
            pluckerMatrix(i,j) = -pluckerMatrix(j,i);

    const Eigen::Matrix<double,4,4> resProjLine = projMat*pluckerMatrix*projMat.transpose();
    const Eigen::Matrix<double,3,1> lineCoeffs = Eigen::Matrix<double,3,1>(resProjLine(2,1),resProjLine(0,2), resProjLine(1,0));
    
    std::cerr << lineCoeffs << std::endl;
    //Compute orthogonal projection of 2D point on a 2D line
    const Eigen::Matrix<double,3,1> line_2d_start = Eigen::Matrix<double,2,1>(m_line_2d_endpoints[0], m_line_2d_endpoints[1]).homogeneous();
    const Eigen::Matrix<double,3,1> line_2d_end = Eigen::Matrix<double,2,1>(m_line_2d_endpoints[2], m_line_2d_endpoints[3]).homogeneous();
    
    
    //We project on m_line_2d_endpoint_start and end on the (finite) line formed by (proj_3d_point_start, proj_3d_point_end)
    double dist_2d_3d_start;
    double dist_2d_3d_end;

    //Step 1: Compute the projection 
    Eigen::Matrix<double,2,1> normalizationCoeff(lineCoeffs(0), lineCoeffs(1));

    dist_2d_3d_start = (lineCoeffs.dot(line_2d_start))/normalizationCoeff.norm();
    dist_2d_3d_end =  (lineCoeffs.dot(line_2d_end))/normalizationCoeff.norm();
    std::cerr << dist_2d_3d_start << " " << dist_2d_3d_end << std::endl;
    double x0 = 0;
    double y0 = (-lineCoeffs(0)*x0-lineCoeffs(2))/lineCoeffs(1);
    double x1 = 1000;
    double y1 = (-lineCoeffs(0)*x1-lineCoeffs(2))/lineCoeffs(1);

    cv::Mat test = cv::imread("/home/victor/Data/Stages/MIT/newer_college_dataset/subset/infra1/subset/"+v->s_Img_path);
    cv::line(test, cv::Point(m_line_2d_endpoints[0], m_line_2d_endpoints[1]), cv::Point(m_line_2d_endpoints[2], m_line_2d_endpoints[3]), cv::Scalar(0,0,255), 2, CV_AA);
    cv::line(test, cv::Point(x0, y0), cv::Point(x1,y1), cv::Scalar(255,0,0), 2, CV_AA);
    cv::imshow("test",test);
    cv::waitKey(0);
}   
}
}