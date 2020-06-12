#include "line_utils.hpp"
namespace openMVG {
namespace sfm {
Eigen::Vector3d Line::getProjection(const Eigen::Matrix4d & projMat) const {
    Eigen::Matrix4d projPlucker = projMat * pluckerMatrix * projMat.transpose();
    Eigen::Vector3d lineCoeff(projPlucker(2,1), projPlucker(0,2), projPlucker(1,0));

        return lineCoeff;
}

float getDistToLine(const Eigen::Vector3f lPoint,
                           const Eigen::Vector3f lDir,
                           const Eigen::Vector3f curPoint)
{
    return ((curPoint - lPoint).cross(lDir)).squaredNorm()/lDir.norm();
}
}
}