#include <Eigen/Geometry>
#include <pcl/ModelCoefficients.h>

namespace Eigen{
    using Vector6f = Matrix<float,6,1>;
    using Vector6d = Matrix<double,6,1>;

}
namespace openMVG {
namespace sfm {
/**
 * Utility function returning the distance of a 3D point to a 3D line (characterized by a point and a direction)
*/
float getDistToLine(const Eigen::Vector3f lPoint,
                           const Eigen::Vector3f lDir,
                           const Eigen::Vector3f curPoint);
class Line{
public:
    Eigen::Matrix4d pluckerMatrix;
    Eigen::Vector6d pluckerVector;

    Eigen::Vector3d getProjection(const Eigen::Matrix4d & projMat) const;

    Line(const Eigen::Vector3d pA, const Eigen::Vector3d pB){
        Eigen::Vector4d A = pA.homogeneous();
        Eigen::Vector4d B = pB.homogeneous();
        pluckerMatrix = A*B.transpose()-B*A.transpose();

    }
    Line(const pcl::ModelCoefficients& line) : Line(Eigen::Vector3d(line.values[0], line.values[1], line.values[2]), Eigen::Vector3d(line.values[0]+line.values[3], line.values[1]+line.values[4], line.values[2]+line.values[5]))
    {

    }

private:
    void setMatrixFromVector(void){
        pluckerMatrix = Eigen::Matrix4d::Zero();
        for(size_t i=0;i<3;++i){
            pluckerMatrix(i,0) = pluckerVector(i);
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


}
}