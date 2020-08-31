#ifndef OPENMVG_SFM_SFM_LINE_UTILS_HPP
#define OPENMVG_SFM_SFM_LINE_UTILS_HPP

#include <Eigen/Geometry>

#include <map>
#include <set>
#define PCL_NO_PRECOMPILE
#include <pcl/pcl_macros.h>
#include <pcl/point_types.h>
#include <pcl/common/common_headers.h>
#include <pcl/ModelCoefficients.h>

#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Polygon_2_algorithms.h>
#include <CGAL/intersections.h>
#include <pcl/point_types.h>
#include <opencv2/opencv.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#include <ceres/rotation.h>
#include <ceres/types.h>
#include "openMVG/numeric/eigen_alias_definition.hpp"
#include "openMVG/sfm/sfm_data.hpp"
#include "third_party/lsd/LineDescriptor.hh"

namespace Eigen{
    using Vector6f = Matrix<float,6,1>;
    using Vector6d = Matrix<double,6,1>;

}
namespace LBD{
using Descriptor = std::vector<float>;
}
namespace PARAMS{
    const double tVertAngle(45*M_PI/180); // used in 
    const double tMergeDeltaAngle(10*M_PI/180); // used in
    const int tMinViewsLine(2); //Minimum number of views for a 3D line to be considered
    const double tOutlierMergeLines(0.7); //70% of the inliers of line A need to be inliers of lineA+lineB
    const double tDeltaAngle3d2d(7*M_PI/180); //max delta between 3d line reprojected and segment
    const double tEdgeSmoothness(0.0);
    const int nNeighborsSmoothness(5);
    const float tMinLength3DSegment(1); 
    const int tOrthoDistLineMerge(3); //max #pixels between 2 // lines for fusion
    const int tMinLenLine(30); //in pixels
    
    /** LIDAR specs (OS1-64) **/
    const int nRings(64);
    const int widthLidar(1024);

    /** Matching **/
    const double tMaxRelativeOverlap(0.3);
    const double tDistanceLinePointMatch(10); 
    const double tDistanceLineCorrespondence(300); 
    const double tDeltaAngleAssociation(10*M_PI/180);
    const float tMaxFeatDistance(0.1);
    const double tOrthoDistMatch(0.5); 

    /** Colormap **/
    static const float r[] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.00588235294117645f,0.02156862745098032f,0.03725490196078418f,0.05294117647058827f,0.06862745098039214f,0.084313725490196f,0.1000000000000001f,0.115686274509804f,0.1313725490196078f,0.1470588235294117f,0.1627450980392156f,0.1784313725490196f,0.1941176470588235f,0.2098039215686274f,0.2254901960784315f,0.2411764705882353f,0.2568627450980392f,0.2725490196078431f,0.2882352941176469f,0.303921568627451f,0.3196078431372549f,0.3352941176470587f,0.3509803921568628f,0.3666666666666667f,0.3823529411764706f,0.3980392156862744f,0.4137254901960783f,0.4294117647058824f,0.4450980392156862f,0.4607843137254901f,0.4764705882352942f,0.4921568627450981f,0.5078431372549019f,0.5235294117647058f,0.5392156862745097f,0.5549019607843135f,0.5705882352941174f,0.5862745098039217f,0.6019607843137256f,0.6176470588235294f,0.6333333333333333f,0.6490196078431372f,0.664705882352941f,0.6803921568627449f,0.6960784313725492f,0.7117647058823531f,0.7274509803921569f,0.7431372549019608f,0.7588235294117647f,0.7745098039215685f,0.7901960784313724f,0.8058823529411763f,0.8215686274509801f,0.8372549019607844f,0.8529411764705883f,0.8686274509803922f,0.884313725490196f,0.8999999999999999f,0.9156862745098038f,0.9313725490196076f,0.947058823529412f,0.9627450980392158f,0.9784313725490197f,0.9941176470588236f,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.9862745098039216f,0.9705882352941178f,0.9549019607843139f,0.93921568627451f,0.9235294117647062f,0.9078431372549018f,0.892156862745098f,0.8764705882352941f,0.8607843137254902f,0.8450980392156864f,0.8294117647058825f,0.8137254901960786f,0.7980392156862743f,0.7823529411764705f,0.7666666666666666f,0.7509803921568627f,0.7352941176470589f,0.719607843137255f,0.7039215686274511f,0.6882352941176473f,0.6725490196078434f,0.6568627450980391f,0.6411764705882352f,0.6254901960784314f,0.6098039215686275f,0.5941176470588236f,0.5784313725490198f,0.5627450980392159f,0.5470588235294116f,0.5313725490196077f,0.5156862745098039f,0.5f};
    static const float g[] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.001960784313725483f,0.01764705882352935f,0.03333333333333333f,0.0490196078431373f,0.06470588235294117f,0.08039215686274503f,0.09607843137254901f,0.111764705882353f,0.1274509803921569f,0.1431372549019607f,0.1588235294117647f,0.1745098039215687f,0.1901960784313725f,0.2058823529411764f,0.2215686274509804f,0.2372549019607844f,0.2529411764705882f,0.2686274509803921f,0.2843137254901961f,0.3f,0.3156862745098039f,0.3313725490196078f,0.3470588235294118f,0.3627450980392157f,0.3784313725490196f,0.3941176470588235f,0.4098039215686274f,0.4254901960784314f,0.4411764705882353f,0.4568627450980391f,0.4725490196078431f,0.4882352941176471f,0.503921568627451f,0.5196078431372548f,0.5352941176470587f,0.5509803921568628f,0.5666666666666667f,0.5823529411764705f,0.5980392156862746f,0.6137254901960785f,0.6294117647058823f,0.6450980392156862f,0.6607843137254901f,0.6764705882352942f,0.692156862745098f,0.7078431372549019f,0.723529411764706f,0.7392156862745098f,0.7549019607843137f,0.7705882352941176f,0.7862745098039214f,0.8019607843137255f,0.8176470588235294f,0.8333333333333333f,0.8490196078431373f,0.8647058823529412f,0.8803921568627451f,0.8960784313725489f,0.9117647058823528f,0.9274509803921569f,0.9431372549019608f,0.9588235294117646f,0.9745098039215687f,0.9901960784313726f,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.9901960784313726f,0.9745098039215687f,0.9588235294117649f,0.943137254901961f,0.9274509803921571f,0.9117647058823528f,0.8960784313725489f,0.8803921568627451f,0.8647058823529412f,0.8490196078431373f,0.8333333333333335f,0.8176470588235296f,0.8019607843137253f,0.7862745098039214f,0.7705882352941176f,0.7549019607843137f,0.7392156862745098f,0.723529411764706f,0.7078431372549021f,0.6921568627450982f,0.6764705882352944f,0.6607843137254901f,0.6450980392156862f,0.6294117647058823f,0.6137254901960785f,0.5980392156862746f,0.5823529411764707f,0.5666666666666669f,0.5509803921568626f,0.5352941176470587f,0.5196078431372548f,0.503921568627451f,0.4882352941176471f,0.4725490196078432f,0.4568627450980394f,0.4411764705882355f,0.4254901960784316f,0.4098039215686273f,0.3941176470588235f,0.3784313725490196f,0.3627450980392157f,0.3470588235294119f,0.331372549019608f,0.3156862745098041f,0.2999999999999998f,0.284313725490196f,0.2686274509803921f,0.2529411764705882f,0.2372549019607844f,0.2215686274509805f,0.2058823529411766f,0.1901960784313728f,0.1745098039215689f,0.1588235294117646f,0.1431372549019607f,0.1274509803921569f,0.111764705882353f,0.09607843137254912f,0.08039215686274526f,0.06470588235294139f,0.04901960784313708f,0.03333333333333321f,0.01764705882352935f,0.001960784313725483f,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    static const float b[] = {0.5f,0.5156862745098039f,0.5313725490196078f,0.5470588235294118f,0.5627450980392157f,0.5784313725490196f,0.5941176470588235f,0.6098039215686275f,0.6254901960784314f,0.6411764705882352f,0.6568627450980392f,0.6725490196078432f,0.6882352941176471f,0.7039215686274509f,0.7196078431372549f,0.7352941176470589f,0.7509803921568627f,0.7666666666666666f,0.7823529411764706f,0.7980392156862746f,0.8137254901960784f,0.8294117647058823f,0.8450980392156863f,0.8607843137254902f,0.8764705882352941f,0.892156862745098f,0.907843137254902f,0.9235294117647059f,0.9392156862745098f,0.9549019607843137f,0.9705882352941176f,0.9862745098039216f,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.9941176470588236f,0.9784313725490197f,0.9627450980392158f,0.9470588235294117f,0.9313725490196079f,0.915686274509804f,0.8999999999999999f,0.884313725490196f,0.8686274509803922f,0.8529411764705883f,0.8372549019607844f,0.8215686274509804f,0.8058823529411765f,0.7901960784313726f,0.7745098039215685f,0.7588235294117647f,0.7431372549019608f,0.7274509803921569f,0.7117647058823531f,0.696078431372549f,0.6803921568627451f,0.6647058823529413f,0.6490196078431372f,0.6333333333333333f,0.6176470588235294f,0.6019607843137256f,0.5862745098039217f,0.5705882352941176f,0.5549019607843138f,0.5392156862745099f,0.5235294117647058f,0.5078431372549019f,0.4921568627450981f,0.4764705882352942f,0.4607843137254903f,0.4450980392156865f,0.4294117647058826f,0.4137254901960783f,0.3980392156862744f,0.3823529411764706f,0.3666666666666667f,0.3509803921568628f,0.335294117647059f,0.3196078431372551f,0.3039215686274508f,0.2882352941176469f,0.2725490196078431f,0.2568627450980392f,0.2411764705882353f,0.2254901960784315f,0.2098039215686276f,0.1941176470588237f,0.1784313725490199f,0.1627450980392156f,0.1470588235294117f,0.1313725490196078f,0.115686274509804f,0.1000000000000001f,0.08431372549019622f,0.06862745098039236f,0.05294117647058805f,0.03725490196078418f,0.02156862745098032f,0.00588235294117645f,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

    const std::string imgExtension = ".png";

}
namespace pcl{

struct EIGEN_ALIGN16 PointXYZIRT{
     PCL_ADD_POINT4D;                    // quad-word XYZ
     float intensity;
     std::uint32_t t;
     std::uint16_t reflectivity;
     std::uint8_t ring;
     std::uint16_t noise;
     std::uint32_t range;
     EIGEN_MAKE_ALIGNED_OPERATOR_NEW     // ensure proper alignment
};
struct EIGEN_ALIGN16 XPointXYZ{
       PCL_ADD_POINT4D;                  // preferred way of adding a XYZ+padding
       float nClustered;
       PCL_MAKE_ALIGNED_OPERATOR_NEW
      // make sure our new allocators are aligned
};                    // enforce SSE padding for correct memory alignment
};


 POINT_CLOUD_REGISTER_POINT_STRUCT(pcl::PointXYZIRT,
                                   (float, x, x)
                                   (float, y, y)
                                   (float, z, z)
                                   (float, intensity, intensity)
                                   (std::uint32_t, t, t)
                                   (std::uint16_t, reflectivity, reflectivity)
                                   (std::uint8_t, ring, ring)
                                   (std::uint16_t, noise, noise)
                                   (std::uint32_t, range, range)
                                   )

 POINT_CLOUD_REGISTER_POINT_STRUCT (pcl::XPointXYZ,
                                    (float, x, x)
                                    (float, y, y)
                                    (float, z, z)
                                    (float, nClustered, nClustered)
                                    )

namespace openMVG {
namespace sfm {
typedef ::CGAL::Exact_predicates_exact_constructions_kernel CGAL_K;

template <typename T>
using PointCloudPtr = typename pcl::PointCloud<T>::Ptr;

using PointCloudXYZ = pcl::PointCloud<pcl::PointXYZ>;
using PointCloudXYZRGB = pcl::PointCloud<pcl::PointXYZRGB>;
using VelodynePointCloud = pcl::PointCloud<pcl::PointXYZIRT>;
using Endpoints2 = std::pair<Vec2, Vec2>;
using Endpoints3 = std::pair<Vec3, Vec3>;

//See https://en.wikipedia.org/wiki/Pl%C3%BCcker_coordinates for additional details
class MyLine{
    public:
        Eigen::Matrix4d pluckerMatrix; //4x4 skew-symmetric matrix
        Eigen::Vector6d pluckerVector; //(L_10, L_20, L_30, L_21, L_31, L_32)
        Eigen::Vector6d pointDirRep; 
public:
        /**
         * Transforms a line from a coordinate frame to another using a SE(3) matrix
         * Returns a new line instance 
        */
        MyLine changeFrame(Mat4 transform) const{
            Vec3 p1 = (transform * pointDirRep.head(3).homogeneous()).hnormalized();
            Vec3 p2 = (transform * (pointDirRep.head(3) + pointDirRep.tail(3)).homogeneous()).hnormalized();
            return MyLine(p1, p2);
        }

        Vec3 getDirection(void) const{
            return Vec3(pluckerMatrix(3,0), pluckerMatrix(3,1), pluckerMatrix(3,2)).normalized();
        }
        /**
         * Projects a 3D line to an image plane
         * The resulting vector corresponds to the 3 coefficients of the 2D line equation
        */
        Vec3 getProjection(const Eigen::Matrix4d & projMat) const {
            Eigen::Matrix4d projPlucker = projMat * pluckerMatrix * projMat.transpose();
            Vec3 lineCoeff(projPlucker(2,1), projPlucker(0,2), projPlucker(1,0));

            return lineCoeff;
        }
        double distanceFromLine(const Vec3& point){
            Vec3 pointOnLine = pointDirRep.head(3); 
            Vec3 res = (pointOnLine - point).cross(getDirection());
            return res.norm();
        }
        // Initialization from 2 3D points
        MyLine(const Vec3 pA, const Vec3 pB){
            Eigen::Vector4d A = pA.homogeneous();
            Eigen::Vector4d B = pB.homogeneous();

            pluckerMatrix = A*B.transpose()-B*A.transpose();
            pointDirRep.head(3) = pA;
            pointDirRep.tail(3) = (pB-pA).normalized();
            setVectorFromMatrix();
        }
        //Initialization from pcl coefficients (6D)
        MyLine(const pcl::ModelCoefficients& line) : MyLine(Vec3(line.values[0], line.values[1], line.values[2]), Vec3(line.values[0]+line.values[3], line.values[1]+line.values[4], line.values[2]+line.values[5]))
        {
            
        }
        MyLine(const Eigen::Vector6d model) {
            pluckerVector = model;
            setMatrixFromVector();

            double normCst = pluckerVector.block(0,0,3,1).norm();
            Eigen::Vector4d closestPointH = pluckerMatrix * pluckerMatrix * Eigen::Vector4d(0,0,0,1); //[L]x[L]x\piinf
            pointDirRep.block(0,0,3,1) = closestPointH.hnormalized();
            pointDirRep.block(3,0,3,1) = getDirection();
        }
    private:
        void setMatrixFromVector(void){
            pluckerMatrix = Eigen::Matrix4d::Zero();
            for(size_t i=1;i<4;++i){
                pluckerMatrix(i,0) = pluckerVector(i-1);
            }
            pluckerMatrix(2,1) = pluckerVector(3);
            pluckerMatrix(3,1) = pluckerVector(4);
            pluckerMatrix(3,2) = pluckerVector(5);

            for(size_t i=0;i<4;++i)
                for(size_t j=i;j<4;++j)
                    pluckerMatrix(i,j) = -pluckerMatrix(j,i);
        }
        void setVectorFromMatrix(void){
            pluckerVector(0) = pluckerMatrix(1,0);
            pluckerVector(1) = pluckerMatrix(2,0);
            pluckerVector(2) = pluckerMatrix(3,0);
            pluckerVector(3) = pluckerMatrix(2,1);
            pluckerVector(4) = pluckerMatrix(3,1);
            pluckerVector(5) = pluckerMatrix(3,2);
        }
    };

// Same as above but finite 3D line: defined by its two 3D endpoints
class Segment3D: public MyLine
{
public:
    Endpoints2 endpoints2D;
    Endpoints3 endpoints3D;
    int view;
    std::vector<LBD::Descriptor> descriptors;    
    /**
     * Computes the L2 distance between 2 line band descriptors (useful for matching)
    */
    double L2Norm(const LBD::Descriptor& v1, const LBD::Descriptor& v2) const{
        double sum = 0;
        for (unsigned int i = 0 ; i < v1.size() ; ++i)
            sum += pow(v1.at(i) - v2.at(i), 2.);
        
        return pow(sum, 1/2.);
    }
    /**
     * Return the length of the 3D segment
    */
    double norm(void) const{
        return (endpoints3D.second - endpoints3D.first).norm();
    }
    /**
     * Computes the distance in feature space between two segments
    **/
    double featureDistance(const Segment3D& other) const{ 
        double minDis, dis;

		short sameLineSize = descriptors.size();
        short sameLineSizeR = other.descriptors.size();

        minDis = 100;
        unsigned int dimOfDes = descriptors.at(0).size();

        for (short lineIDInSameLines = 0; lineIDInSameLines < sameLineSize; lineIDInSameLines++)
        {
            for (short lineIDInSameLinesR = 0; lineIDInSameLinesR < sameLineSizeR; lineIDInSameLinesR++)
            {
                dis = L2Norm(descriptors.at(lineIDInSameLines), other.descriptors.at(lineIDInSameLinesR));
                if (dis < minDis)
                {
                    minDis = dis;
                }
            }
        } 
        return minDis;
    }
    /**
     * The following two functions are utility functions for debugging purposes
    **/
    void debug(int id) const {
        std::cerr << "~~~~~ Segment #" << id << " ~~~~~" << std::endl;
        std::cerr << "View: " << view << std::endl;
        std::cerr << "Endpoints in view: " << "(" << endpoints2D.first(0) << ";" << endpoints2D.first(1) << "), (" << endpoints2D.second(0) << ";" << endpoints2D.second(1) << ")" << std::endl;
        std::cerr << "Endpoints in WF: " << "(" << endpoints3D.first(0) << ";" << endpoints3D.first(1) << ";" <<  endpoints3D.first(2) <<"), (" <<
         endpoints3D.second(0) << ";" << endpoints3D.second(1) << ";" << endpoints3D.second(2) << ")" << std::endl;
        std::cerr << "Norm: " << this->norm() << std::endl;
        std::cerr << " ~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
    }
    void visualize(const Eigen::Matrix4d projection, const cv::Mat& img, int id) const {
        debug(id);
        cv::line(img, cv::Point(endpoints2D.first(0), endpoints2D.first(1)), cv::Point(endpoints2D.second(0), endpoints2D.second(1)), cv::Scalar(255,0,0));
        MyLine test(endpoints3D.first, endpoints3D.second);
        Vec3 backProjLine = test.getProjection(projection);
        cv::Point p1, p2;
        p1.y = 0;
        p2.y = 1000;
        p1.x = (-backProjLine(1)*p1.y-backProjLine(2))/backProjLine(0);
        p2.x = (-backProjLine(1)*p2.y-backProjLine(2))/backProjLine(0);
        Vec3 reprojA = (projection * endpoints3D.first.homogeneous()).hnormalized();
        Vec3 reprojB = (projection * endpoints3D.second.homogeneous()).hnormalized();
        
        cv::circle(img, cv::Point(reprojA(0), reprojA(0)), 1, cv::Scalar(0,0,255));
        cv::circle(img, cv::Point(reprojB(1), reprojB(1)), 1, cv::Scalar(0,0,255));

        cv::line(img, p1, p2, 1);

        cv::imshow("Debug segment", img);
        cv::waitKey(0);
    }
    //Constructor
    Segment3D(const pcl::ModelCoefficients& line, Endpoints2& endpoints_, Endpoints3& endpoints2_, int view_,
      std::vector<LBD::Descriptor> descs):MyLine(line), endpoints2D(endpoints_), endpoints3D(endpoints2_),view(view_), descriptors(descs){}

    bool operator<(const Segment3D& other) const {
        return norm() < other.norm();
    }


};

double getCoeffLine(const Vec2& poi,
                    const Vec2& normal,
                    const Vec2& refPoint);

double pointToLineDist(const Vec2& point,
                       const Endpoints2& line);


void projectPoint2Line2D(const Vec2& curPoint,
                         const Vec2& pointLine,
                         const Vec2& normalizedDirLine,
                         Vec2& nPoint);

double distPoint2Line2D(const Vec2& curPoint,
                        const Vec2& pointLine,
                        const Vec2& dirLine);

double distPoint2Segment(const Vec2& point,
                         const Endpoints2& endpoints,
                         const Vec2 normal);

float getParameter3(const Vec3& point,
                    const Vec3& refPoint,
                    const MyLine& lineModel);

Vec2 equation2Normal2D(const Vec3& eqn);

Vec2 endpoints2Normal(Endpoints2 endpoints);

double CLIP_ANGLE(double angle);

double angleBetweenLines(const Endpoints2& lineA,
                         const Endpoints2& lineB);

bool segmentComp(std::pair<int, Segment3D>& p1, std::pair<int, Segment3D>& p2);

// Union-find
int root(int i, std::vector<int>& clustersSegment);
void joinSets(int i1, int i2,  std::vector<int>& clustersSegment, int * rank);

Eigen::Matrix4d convertRTEigen(const geometry::Pose3& pose);

Vec3 projectC2I(const Vec3& pt, const Mat3& K);
Vec3 projectW2I(const Vec3& pt, const Mat34& projMatrix);

void ShowManyImages(std::string title, int nArgs, std::vector<cv::Mat>& imgs) ;

void PRINT_VECTOR(const Eigen::VectorXf vector, int n);

Vec3 getEndpointLocation(const MyLine& l,
                         const Vec2 pt,
                         const Mat3& K);
}
}
#endif
