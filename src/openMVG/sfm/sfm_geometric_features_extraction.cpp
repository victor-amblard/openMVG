#include "sfm_geometric_features_extraction.hpp"
#include <fstream>
#include <cstdlib>

#ifdef OPENMVG_USE_OPENMP
#include <omp.h>
#endif

namespace openMVG {
namespace sfm {

void computeLinesNDescriptors(const std::string& filename,
                              std::vector<Endpoints2>& allLinesImage,
                              std::vector<std::vector<LBD::Descriptor>>& allDescriptors)
{
    cv::Mat img = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    LineDescriptor lineDesc;

    ScaleLines linesInImage;
    lineDesc.GetLineDescriptor(img, linesInImage);
    std::cerr << linesInImage.size() << " lines detected (" << filename << ")" <<  std::endl;
    for (auto & line: linesInImage){
        float startPointX = line[0].startPointX;
        float startPointY = line[0].startPointY;
        float endPointX = line[0].endPointX;
        float endPointY = line[0].endPointY;
        
        // Remove short line segments 
        if (pow((endPointY - startPointY),2.) + pow((endPointX - startPointX),2.) < pow(PARAMS::tMinLenLine,2.))
            continue;

        auto endpoints2d = std::make_pair(Vec2(startPointX, startPointY), Vec2(endPointX, endPointY));
        allLinesImage.push_back(endpoints2d);
        std::vector<LBD::Descriptor> curDescriptors(line.size());
        for (unsigned int i = 0 ; i < line.size() ; ++i)
            curDescriptors.at(i) = line[i].descriptor;
        
        allDescriptors.push_back(curDescriptors);
    }
}
//Currently not used
void processLinesLSD(const std::string& imgFilename,
                     std::vector<Endpoints2>& allLinesImage){

    const double tDeltaAngle(10 * M_PI / 180.);
    const double tDistEndpoints(20);
    const double tDistOrthoEndpoints(5);
    const double tDeltaL(1.3);
    const double tMinLineFinal(70.);
    const int nIt(1);

    cv::Mat srcImg = cv::imread(imgFilename);

    std::vector<Endpoints2> tmpBuffer;
    tmpBuffer.assign(allLinesImage.begin(), allLinesImage.end());

    for (auto i=0;i<tmpBuffer.size();++i){
        auto elem = tmpBuffer.at(i);
        cv::line(srcImg, cv::Point(elem.first(0), elem.first(1)), cv::Point(elem.second(0), elem.second(1)), cv::Scalar(255,0,0));
    }

    // cv::imshow("Raw image", srcImg);
    // cv::waitKey(0);

    std::vector<Endpoints2> processBuffer;
    std::vector<bool> seen;
    
    for (auto it=0;it<1;++it){
        std::cerr << " ----- ----- ----- " << std::endl;
        seen.reserve(tmpBuffer.size());
        for(int i=0 ; i < tmpBuffer.size(); ++i)
            seen.push_back(false);  
        
        allLinesImage = tmpBuffer;

        for (size_t i1 = 0;i1 < tmpBuffer.size();++i1){
            auto endpoints1 = tmpBuffer.at(i1);
            auto n1 = endpoints2Normal(endpoints1);
            
            for(size_t i2 = i1+1;i2 < tmpBuffer.size();++i2){
                if (seen.at(i2))
                    continue;
                auto endpoints2 = tmpBuffer.at(i2);

                double angle = angleBetweenLines(endpoints1, endpoints2);

                // If angle < thresh angle
                if (angle < tDeltaAngle){
                    
                    Vec2 ort(-n1(1), n1(0));
                    bool allDists[4] = {((endpoints1.first-endpoints2.first).norm() < tDistEndpoints) && (std::fabs((endpoints1.first-endpoints2.first).dot(ort)) < tDistOrthoEndpoints),
                                        ((endpoints1.first-endpoints2.second).norm() < tDistEndpoints)&& (std::fabs((endpoints1.first-endpoints2.second).dot(ort)) < tDistOrthoEndpoints),
                                        ((endpoints1.second-endpoints2.first).norm() < tDistEndpoints) &&  (std::fabs((endpoints1.second-endpoints2.first).dot(ort)) < tDistOrthoEndpoints),
                                        ((endpoints1.second-endpoints2.second).norm() < tDistEndpoints) &&  (std::fabs((endpoints1.second-endpoints2.second).dot(ort)) < tDistOrthoEndpoints)};

                    if (allDists[0] || allDists[1] || allDists[2] || allDists[3]){
                        bool valid = false;
                        if ((allDists[0] && allDists[3]) || (allDists[1] && allDists[2])){ //parallel lines
                            //We keep the longest line
                            double l1 = (endpoints1.second-endpoints1.first).norm();
                            double l2 = (endpoints2.second-endpoints2.first).norm();
                            double deltaL = std::max(l1, l2)/ std::min(l1,l2);
                            if (deltaL > tDeltaL){
                                continue;
                            }
                            if (l1 > l2){
                                processBuffer.push_back(tmpBuffer.at(i1));
                                valid = true;
                            }else{
                                processBuffer.push_back(tmpBuffer.at(i2));
                                valid = true;
                            }
                        }else{ //longer lines
                            if (allDists[0] && !allDists[1] && !allDists[2] && !allDists[3]){
                                processBuffer.push_back(std::make_pair(endpoints1.second, endpoints2.second));
                                valid = true;
                            }if (allDists[1] && !allDists[0] && !allDists[2] && !allDists[3]){
                                processBuffer.push_back(std::make_pair(endpoints1.second, endpoints2.first));
                                valid = true;
                            }if (allDists[2] && !allDists[0] && !allDists[1] && !allDists[3]){
                                processBuffer.push_back(std::make_pair(endpoints1.first, endpoints2.second));
                                valid = true;
                            }if (allDists[3] && !allDists[1] && !allDists[2] && !allDists[0]){
                                processBuffer.push_back(std::make_pair(endpoints1.first, endpoints2.first));
                                valid = true;
                            }
                        }
                        if (valid){
                            seen.at(i2) = true;
                            seen.at(i1) = true;
                        }
                        break;
                    }
                }
            }
            if (!seen.at(i1))
                processBuffer.push_back(tmpBuffer.at(i1));


        }
        seen.clear();
        tmpBuffer.clear();
        tmpBuffer = processBuffer;
        processBuffer.clear();
      }
      
    allLinesImage.clear();
    for (auto elem: tmpBuffer){
        if ((elem.first-elem.second).norm() > tMinLineFinal){
            allLinesImage.push_back(elem);
        }
    }
    cv::Mat debug = cv::Mat(srcImg.size().height, srcImg.size().width, CV_8UC3, cv::Scalar(255,255,255));

    for (auto i=0;i<allLinesImage.size();++i){
        auto elem = allLinesImage.at(i);
        cv::line(srcImg, cv::Point(elem.first(0), elem.first(1)), cv::Point(elem.second(0), elem.second(1)), cv::Scalar(0,0,255));
        cv::line(debug, cv::Point(elem.first(0), elem.first(1)), cv::Point(elem.second(0), elem.second(1)), cv::Scalar(0,0,0),2);

    }
    // cv::imshow("Modified image", debug);
    // cv::waitKey(0);

}
/**
 * Give a 2D line equation and the dimensions of an image,
 * returns the location of the 2D endpoints in the image
 * ---+----------------
 * |   \              |
 * |    \             |
 * |     \            |
 * |      \           |
 * --------+-----------
 **/
bool getBoundPoints(const Vec3& lineProj,
                  const int width,
                  const int height,
                  Endpoints2& projPoints)
{

    CGAL_K::Point_2 topleftImage((CGAL_K::RT)(0), (CGAL_K::RT)(0));
    CGAL_K::Point_2 bottomRightImage((CGAL_K::RT)width, (CGAL_K::RT)height);

    CGAL_K::Iso_rectangle_2 imageBounds(topleftImage, bottomRightImage);
    CGAL_K::Line_2 line((CGAL_K::RT)lineProj.x(), (CGAL_K::RT)lineProj.y(), (CGAL_K::RT)lineProj.z());

    auto resultInter = CGAL::intersection(imageBounds, line);
    if (resultInter){
        if (const CGAL_K::Point_2* p = boost::get<CGAL_K::Point_2>(&*resultInter)){
            return false;
        }else{
            const CGAL_K::Segment_2* r = boost::get<CGAL_K::Segment_2>(&*resultInter);
                projPoints.first = Eigen::Vector2d(CGAL::to_double(r->source().x()), CGAL::to_double(r->source().y()));
                projPoints.second = Eigen::Vector2d(CGAL::to_double(r->target().x()), CGAL::to_double(r->target().y()));
                return true;
            }
    }else{
       return false;
    }
}

/**
 * Function that take a 3D segment, and computes the views in which 
 * the segment can be considered to be in the field of view
 *  1) Project the segment in the image plane using K and sfm_data 
 *  2) Use the above function to compute its 2D endpoints in the image plane
 *  3) If the segment is partially or entirely in the image, we consider that 
 * the segment is in the field of view
**/
std::vector<std::pair<int, bool>> getViewsSegment(const Segment3D& segment,
                                                  const SfM_Data& sfm_data,
                                                  const Mat3& K)
{

    std::vector<std::pair<int,bool>> result;

     for (const auto& view_it: sfm_data.GetViews()){

        const View * view = view_it.second.get();
        
        if (!sfm_data.IsPoseAndIntrinsicDefined(view))
            continue;
        
        const Pose3& transform = sfm_data.GetPoseOrDie(view_it.second.get());
        const Eigen::Matrix4d EigTransform = convertRTEigen(transform);
        //Projection matrix (world frame --> image plane)
        geometry::Pose3 pose = sfm_data.GetPoseOrDie(view);
        Mat34 projMatrix;
        openMVG::P_From_KRt(K, pose.rotation(), pose.translation(), &projMatrix);
        Mat4 projMatrix4 = Mat4::Identity();
        projMatrix4.block(0,0,3,4) = projMatrix;


        MyLine segmentW(segment.endpoints3D.first, segment.endpoints3D.second);
        Vec3 backProjLine = segmentW.getProjection(projMatrix4);
        Vec2 normal = equation2Normal2D(backProjLine);


        Vec3 backProjStart = projectW2I(segment.endpoints3D.first, projMatrix);
        Vec3 backProjEnd = projectW2I(segment.endpoints3D.second, projMatrix);

        //First thing to check: if infinite line's projection lies inside the image
        Endpoints2 projBoundPoints;
        bool isInFov = getBoundPoints(backProjLine, view->ui_width, view->ui_height, projBoundPoints);
        if (!isInFov || (backProjStart.z() < 0 && backProjEnd.z() < 0))
            continue;

        //Second check: Is the finite segment visible in the image
        double tA = 0;
        double tB = getCoeffLine(backProjEnd.hnormalized(), normal, backProjStart.hnormalized());
        double tStart = getCoeffLine(projBoundPoints.first, normal, backProjStart.hnormalized());
        double tEnd = getCoeffLine(projBoundPoints.second, normal, backProjStart.hnormalized());

        if (tEnd < tStart)
            std::swap(tStart, tEnd);
        if (tB < tA)
           std::swap(tA, tB);
        //Valid if tA <= tStart <= tEnd <= tB (both endpoints outside image) or tA <= tStart <= tB <= tEnd (1 endpoint outside) or tStart <= tA <= tB <= tEnd (both in image)
        if (!(tA <= tStart && tB <= tStart) && !(tA >= tEnd && tB >= tEnd)){
            if ((tA <= tStart && tB >= tEnd) || (tA <= tStart && tB <= tEnd)){ //partial visibility
                result.push_back(std::make_pair(view->id_view, false));
            }else{ //entirely visible
                result.push_back(std::make_pair(view->id_view, true));
            }
        }
    }

    return result;
}
/**
 * Given 2 3D segments: do these segments correspond to different 
 * views of the same actual real-world segment?
 * 4 criteria are used to address this question
 **/
std::pair<bool,Eigen::Vector4f> isMatched(const Segment3D& curSegment,
                                        const Segment3D& refSegment,
                                        const Pose3& transformWF,
                                        const int w,
                                        const int h,
                                        bool completeVisibility,
                                        const Mat3& K){
    
    Mat34 proj2Ref;
    openMVG::P_From_KRt(K, transformWF.rotation(), transformWF.translation(), &proj2Ref);
    Eigen::Matrix4d proj2Ref4 = Eigen::Matrix4d::Identity();
    proj2Ref4.block(0,0,3,4) = proj2Ref;

    //Criterion #1: Angle
    MyLine curSegmentW(curSegment.endpoints3D.first, curSegment.endpoints3D.second);
    Vec3 abcCur = curSegmentW.getProjection(proj2Ref4);
    Vec2 normalCur = Vec2(-abcCur(1), abcCur(0)).normalized();
    Vec2 normalRef = endpoints2Normal(refSegment.endpoints2D);
    
    float relTheta = CLIP_ANGLE(std::acos(normalCur.dot(normalRef))) / PARAMS::tDeltaAngleAssociation;

    // Criterion #2: Distance
    Vec3 reprojStart = projectW2I(curSegment.endpoints3D.first, proj2Ref);
    Vec3 reprojEnd = projectW2I(curSegment.endpoints3D.second, proj2Ref);

    Endpoints2 endpointsCur = std::make_pair(reprojStart.hnormalized(), reprojEnd.hnormalized());

    double totalDistance  = distPoint2Line2D((endpointsCur.first+endpointsCur.second)/2, refSegment.endpoints2D.first, normalRef);
    totalDistance /= PARAMS::tDistanceLineCorrespondence;

    // Criterion #3: Overlap (in case of complete visibility)
    double normCurSegment = (endpointsCur.second - endpointsCur.first).norm();
    double normRefSegment = (curSegment.endpoints2D.second - curSegment.endpoints2D.first).norm();
    double overlap = std::fabs(normCurSegment - normRefSegment) / normRefSegment;


    // Criterion #4: Distance in feature space
    float distFeature = refSegment.featureDistance(curSegment) / PARAMS::tMaxFeatDistance;
    
    // Criterion #5 (anti-occlusion)
    Vec3 dirRef3 = Vec3(refSegment.endpoints3D.second - refSegment.endpoints3D.first).normalized();
    Vec3 vecDist3((curSegment.endpoints3D.first + curSegment.endpoints3D.second)/2 - (refSegment.endpoints3D.first + refSegment.endpoints3D.second)/2);
    double orthoDist = vecDist3.norm () * sqrt(1 - pow(vecDist3.dot(dirRef3)/vecDist3.norm(),2.))/PARAMS::tOrthoDistMatch;
    // std::cerr << "3D distance between centers: " << vecDist3.norm() << ", " << vecDist3.dot(dirRef3)/vecDist3.norm()<< " " << orthoDist << std::endl;
    // PRINT_VECTOR(vecDist3, 3)
    // PRINT_VECTOR(dirRef3, 3)
    bool valid;

    Eigen::Vector4f score(totalDistance, relTheta, distFeature, orthoDist);
    valid = (score(0) < 1) && (score(1) < 1) && (score(2) < 1) && (score(3) < 1);// && (!completeVisibility || (overlap > PARAMS::tMaxRelativeOverlap));

    return std::make_pair(valid, score) ;
}

//Checked
void findCorrespondencesAcrossViews(const std::vector<std::string>& filenames,
                                    std::vector<std::pair<int, Segment3D>>& allSegments,
                                    const Hash_Map<IndexT, std::vector<int>>& segmentsInView,
                                    const SfM_Data& sfm_data,
                                    const Mat3& K,
                                    std::vector<std::vector<int>>&finalLines,
                                    std::map<int, int>& mapSegment)
{

    int nViews = sfm_data.GetViews().size();
    const float MAX_SCORE = 100;
    const float EPS_F = 0.0001;

    std::cerr << "There are " << allSegments.size() << " 3D segments in the dataset" << std::endl;

    std::sort(allSegments.begin(), allSegments.end(), segmentComp); //sort by desc length
    std::vector<std::vector<int>> lineClusters;
    std::vector<std::vector<int>> setIdsByView(sfm_data.GetViews().size());
    Hash_Map<int, DataAssociation> mapDataAssociations;
    std::vector<std::vector<int>> allDataAssociations(allSegments.size());

    std::vector<int> clustersSegment(allSegments.size());
    int rank[10000];
    for (unsigned int i=0;i<allSegments.size();++i){
        clustersSegment.at(i) = i;
        rank[i] = 1;
    }
    
    int curIdxDA(0);

    for (unsigned int iSegment = 0; iSegment < allSegments.size() ; ++iSegment)
        mapSegment[allSegments.at(iSegment).first] = iSegment;

    for(unsigned int iSegment = 0; iSegment < allSegments.size();++iSegment)
    {
        int curIdSegment = allSegments.at(iSegment).first;
        const Segment3D& curSegment(allSegments.at(iSegment).second);
        curSegment.debug(curIdSegment);

        std::vector<std::pair<int, bool>> validViewsReprojection = getViewsSegment(curSegment, sfm_data, K);

        std::cerr << "View # " ;
        if (std::find(validViewsReprojection.begin(), validViewsReprojection.end(), std::make_pair(curSegment.view, true)) == validViewsReprojection.end())
            std::cerr << "Error ! My view is not in the set of visible views!" << std::endl;
        for (unsigned int iView = 0 ; iView < validViewsReprojection.size();++iView)
        {
            int nView = validViewsReprojection.at(iView).first;
            std::cerr << nView << " ";
            const int width = sfm_data.GetViews().at(curSegment.view).get()->ui_width;
            const int height = sfm_data.GetViews().at(curSegment.view).get()->ui_height;

            if (nView != curSegment.view) //We want to avoid to have twice the same view in the same set
            {
                std::pair<float, Eigen::Vector4f> minScore = std::make_pair(MAX_SCORE, Eigen::Vector4f(0,0,0,0));

                int minSegmentId = 0;
                bool match = false;

                for (unsigned int iPotentialMatch = 0; iPotentialMatch < segmentsInView.at(nView).size() ; ++iPotentialMatch)
                {
                    const View * viewP = sfm_data.GetViews().at(nView).get();
                    const Pose3& curTF = sfm_data.GetPoseOrDie(viewP);
                    int segmentId = segmentsInView.at(nView).at(iPotentialMatch);
                    auto score = isMatched(allSegments.at(mapSegment[segmentId]).second, curSegment, curTF, 
                    width, height, validViewsReprojection.at(iView).second, K);

                    // PRINT_VECTOR(score.second,4);
                    if (score.first)
                    {
                        match = true;
                        std::cerr << "Match! (" << curIdSegment << "," << segmentId << ")" << std::endl;
                        if (score.second.norm() < minScore.first)
                        {
                            minScore = std::make_pair(score.second.norm(), score.second);
                            minSegmentId = segmentId;
                        }
                    }
                }

                if (match){
                    std::cerr << curSegment.featureDistance(allSegments.at(mapSegment[minSegmentId]).second) << std::endl;
                    std::cerr << "Segment " << curIdSegment << "(" << curSegment.view << ") corresponds to segment " << minSegmentId << "(" << nView
                    << ")" << std::endl;
                    // #pragma omp critical
                    joinSets(curIdSegment, minSegmentId, clustersSegment, rank);
                    DataAssociation da(curIdxDA, curIdSegment, minSegmentId, minScore.second);
                    allDataAssociations.at(curIdSegment).push_back(curIdxDA);
                    mapDataAssociations.insert({curIdxDA, da});
                    ++curIdxDA;
                    /** 
                    const cv::Mat img = cv::imread(sfm_data.s_root_path+"/"+sfm_data.GetViews().at(curSegment.view)->s_Img_path);
                    const cv::Mat img2 = cv::imread(sfm_data.s_root_path+"/"+sfm_data.GetViews().at(allSegments.at(mapSegment[minSegmentId]).second.view)->s_Img_path);
                    cv::line(img, cv::Point(curSegment.endpoints2D.first(0), curSegment.endpoints2D.first(1)), cv::Point(curSegment.endpoints2D.second(0), curSegment.endpoints2D.second(1)), cv::Scalar(0,0,255),2);
                    cv::line(img2, cv::Point(allSegments.at(mapSegment[minSegmentId]).second.endpoints2D.first(0), allSegments.at(mapSegment[minSegmentId]).second.endpoints2D.first(1)), cv::Point(allSegments.at(mapSegment[minSegmentId]).second.endpoints2D.second(0), allSegments.at(mapSegment[minSegmentId]).second.endpoints2D.second(1)), cv::Scalar(0,0,255),2);
                    
                    cv::putText(img, "Img "+std::to_string(minScore.second(0)), cv::Point(30,30), 3, 1, cv::Scalar(0,0,255), 3);
                    cv::putText(img, "Img "+std::to_string(minScore.second(1)), cv::Point(30,100), 3, 1, cv::Scalar(0,0,255), 3);
                    cv::putText(img, "Img "+std::to_string(minScore.second(2)), cv::Point(30,150), 3, 1, cv::Scalar(0,0,255), 3);
                    cv::putText(img, "Img "+std::to_string(minScore.second(3)), cv::Point(30,200), 3, 1, cv::Scalar(0,0,255), 3);
                    // cv::imshow("["+std::to_string(curIdSegment)+"] - View # "+std::to_string(curSegment.view), img);
                    std::vector<cv::Mat> matchVec = {img, img2};
                    ShowManyImages("DA", 2, matchVec);
                    // **/ 
                }
            }
        
        }
    }
    Hash_Map<int, int> daIdx2ConsecutiveIdx;
    Hash_Map<int, int> daIdx2ConsecutiveIdxR;
    int countKey(0);
    int * allObs = new int [nViews];
    for (int i = 0 ; i < nViews;++i)
        allObs[i] = 0;

    for (int iView = 0 ; iView < nViews ; ++iView){
        for (auto& elem: mapDataAssociations){
            const DataAssociation& da(elem.second);
            
            int key = da.idSegmentA;    
            if (allSegments.at(mapSegment[key]).second.view == iView && daIdx2ConsecutiveIdx.find(key) == daIdx2ConsecutiveIdx.end()){
                daIdx2ConsecutiveIdx[key] = countKey;
                daIdx2ConsecutiveIdxR[countKey] = key;
                ++countKey;
                allObs[iView] ++;
            }
            int key2 = da.idSegmentB;
            if (allSegments.at(mapSegment[key2]).second.view == iView && daIdx2ConsecutiveIdx.find(key2) == daIdx2ConsecutiveIdx.end()){
                daIdx2ConsecutiveIdx[key2] = countKey;
                daIdx2ConsecutiveIdxR[countKey] = key2;
                ++countKey;
                allObs[iView] ++;

            }

        }
    }
    std::ofstream fileO;
    fileO.open("/home/victor/Data/Stages/MIT/clear/test.txt");
    fileO << std::to_string(countKey) << "\n"; 
    fileO << std::to_string(sfm_data.GetViews().size()) << "\n"; 
    int totalDA = curIdxDA; 

    for (int i = 0 ; i < nViews ; ++i)
        fileO << std::to_string(allObs[i]) << " "; 

    fileO << "\n";
    for (auto& elem: mapDataAssociations){
        const DataAssociation& da(elem.second);

        int n1 = daIdx2ConsecutiveIdx[da.idSegmentA];
        int n2 = daIdx2ConsecutiveIdx[da.idSegmentB];
        fileO << std::to_string(n1) << " " << std::to_string(n2) << "\n";
    }
    fileO.close();  

    std::system("cd /home/victor/Data/Stages/MIT/clear/CLEAR_Python/ && rm -f ../test_output.txt && python3 readFromFile.py");

    for (unsigned int i=0;i<allSegments.size();++i){
        clustersSegment.at(i) = i;
        rank[i] = 1;
    }
    std::string line;
    ifstream myfile ("/home/victor/Data/Stages/MIT/clear/test_output.txt");
    if (myfile.is_open())
    {
        while(std::getline(myfile, line)){

            std::stringstream  lineStream(line);

            int curA, curB;
            lineStream >> curA;
            lineStream >> curB;
            int eLeft = daIdx2ConsecutiveIdxR[curA];
            int eRight = daIdx2ConsecutiveIdxR[curB];

            joinSets(eLeft, eRight, clustersSegment, rank);
        }
        myfile.close();
    }

    finalLines = std::vector<std::vector<int>> (allSegments.size());
    for (unsigned int i = 0; i < clustersSegment.size(); ++i){
        int idRoot = root(i, clustersSegment);
        finalLines.at(idRoot).push_back(i);
    }
    
    for (auto it = finalLines.begin(); it!=finalLines.end();){
        std::cerr << "Cur line has " << it->size() << " views in sight" << std::endl;
        if (it->size() >= PARAMS::tMinViewsLine)
            ++it;
        else
            it = finalLines.erase(it);

    }
    std::cerr << "After fusion and thresholding there are " << finalLines.size() << " lines found" << std::endl;

}



void testLineReprojectionCostFunction(const double * const cam_intrinsics,
                                      const double * const cam_extrinsics,
                                      const double * const line_3d_endpoint,
                                      const double * m_line_2d_endpoints,
                                      const SfM_Data& sfm_data,
                                      const int idView)
{
    View * v = sfm_data.GetViews().at(idView).get();

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

    cv::Mat test = cv::imread(sfm_data.s_root_path+"/"+v->s_Img_path);
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
                                      const SfM_Data& sfm_data,
                                      const int idView)
{   
    View * v = sfm_data.GetViews().at(idView).get();

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
    std::cerr << sfm_data.s_root_path+"/"+v->s_Img_path << std::endl;
    cv::Mat test = cv::imread(sfm_data.s_root_path+"/"+v->s_Img_path);
    cv::line(test, cv::Point(m_line_2d_endpoints[0], m_line_2d_endpoints[1]), cv::Point(m_line_2d_endpoints[2], m_line_2d_endpoints[3]), cv::Scalar(0,0,255), 2, CV_AA);
    cv::line(test, cv::Point(x0, y0), cv::Point(x1,y1), cv::Scalar(255,0,0), 2, CV_AA);
    cv::imshow("test",test);
    cv::waitKey(0);
}   


void group3DLines(const std::vector<std::pair<int, Segment3D>>& allSegments,
                const std::map<int, int>& mapSegment,
                std::vector<std::vector<int>>& finalLines,
                Hash_Map<IndexT, MyLine>& allLines)
{

    // RANSAC parameters
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_LINE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(2);
    seg.setMaxIterations(50);
    int i(0);
    for (auto it=finalLines.begin();it!=finalLines.end();){
        PointCloudXYZ::Ptr curLinePoints(new PointCloudXYZ);
        std::vector<int> resultInliers(it->size());
        for (unsigned int iSegment = 0 ; iSegment < it->size() ; ++iSegment){

            const Segment3D& curSegment = allSegments.at(mapSegment.at((*it)[iSegment])).second;
            resultInliers.at(iSegment) = 0;
            curLinePoints->push_back(pcl::PointXYZ(curSegment.endpoints3D.first.x(), curSegment.endpoints3D.first.y(), curSegment.endpoints3D.first.z()));
            curLinePoints->push_back(pcl::PointXYZ(curSegment.endpoints3D.second.x(), curSegment.endpoints3D.second.y(), curSegment.endpoints3D.second.z()));
        }

        // Perform RANSAC to get line equation, which will be further optimized during optimization

        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices());

        seg.setInputCloud(curLinePoints);
        seg.segment(*inliers, *coefficients);
        const MyLine  ln(*coefficients);     

        if (inliers->indices.size() > 6){ //3 views
            
            for (auto inlier : inliers->indices){
                int idInlierSegment = static_cast<int>(std::floor(inlier/2.f));
                resultInliers.at(idInlierSegment) += 1;
            }
            std::vector<int> toBeRemoved;
            for (unsigned int iInlier = 0 ; iInlier < it->size() ; ++iInlier){
                if (resultInliers.at(iInlier) != 2){
                    toBeRemoved.push_back((*it)[iInlier]);
                }
            }
            std::cerr << "Will remove " << toBeRemoved.size() << " of the " << it->size() << " segments in the current cluster" << std::endl;
            for (auto itSegment = it->begin();itSegment != it->end();){
                if (std::find(toBeRemoved.begin(), toBeRemoved.end(), *itSegment) != toBeRemoved.end()){
                    itSegment = it->erase(itSegment);
                }else{
                    ++itSegment;
                }
            }
            allLines.insert({i, ln});
            ++i;
            ++it;
        }else{
            it = finalLines.erase(it);
        }

    }
}
/**
 * Given a new line model (derived from the optimization) in world frame and a set of 2D segments associated to that line
 * Returns the two 3D most likely line endpoints in world frame and all the views in which the segment is seen
**/ 
std::pair<Endpoints3, std::vector<int>> extractFinalEndpoints(const Eigen::Vector6d& lineModel,
                                                            const std::vector<std::pair<int, Segment3D>>& allSegments,
                                                            const std::vector<int>& segmentIds,
                                                            std::map<int, int>& mapIdx,
                                                            const Mat3& K,
                                                            const SfM_Data& sfmData)

{
    MyLine curLine(lineModel);

    float tMin = std::numeric_limits<float>::max();
    float tMax = -std::numeric_limits<float>::max();

    // Since all poses have changed, we need to recompute the 3D endpoints
    // The new endpoint will be the closest point to the intersection between the epipolar line and the infinite line
    int i(0);
    Vec3 refPoint;
    std::vector<int> views;
    views.reserve(segmentIds.size());

    for (const auto& lineCorrespondence: segmentIds)
    {
        IndexT idxSegment = lineCorrespondence;
        const Segment3D & seg = allSegments.at(mapIdx[idxSegment]).second;
        const geometry::Pose3& curPose = sfmData.GetPoseOrDie(sfmData.GetViews().at(seg.view).get());
        Mat4 world2camera = Mat4::Identity();
        world2camera.block(0,0,3,3) = curPose.rotation();
        world2camera.block(0,3,3,1) = curPose.translation();
        
        MyLine cameraLine = curLine.changeFrame(world2camera); //3d line in camera frame

        Vec2 sEndpoint2 = seg.endpoints2D.first;
        Vec2 eEndpoint2 = seg.endpoints2D.second;

        Vec3 sEndpoint3C  = getEndpointLocation(cameraLine, sEndpoint2, K); //get closest point in camera frame to the line
        Vec3 eEndpoint3C  = getEndpointLocation(cameraLine, eEndpoint2, K);

        Vec3 sEndpoint3  = (world2camera.inverse() * sEndpoint3C.homogeneous()).head(3); // camera --> world frame
        Vec3 eEndpoint3  = (world2camera.inverse() * eEndpoint3C.homogeneous()).head(3);

          if (i == 0){
            refPoint = sEndpoint3;
          }
          float tS = getParameter3(sEndpoint3, refPoint, curLine); //Line's parametrization
          float tE = getParameter3(eEndpoint3, refPoint, curLine);

          tMin = std::min(std::min(tMin, tS), tE);
          tMax = std::max(std::max(tMax, tS), tE);

          ++i;
          views.push_back(seg.view);
    }

    Vec3 mEndpoint = refPoint + tMin * curLine.getDirection();
    Vec3 MEndpoint = refPoint + tMax * curLine.getDirection();

    return std::make_pair(std::make_pair(mEndpoint, MEndpoint), views);
}
}
}