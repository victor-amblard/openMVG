#include "openMVG/sfm/line_utils.hpp"

#include "testing/testing.h"

//test getDistToLine
float threshold(0.0001f);

TEST(get_dist_line, point_on_line){
    Eigen::Vector3f pol(1,1,1);
    Eigen::Vector3f pl(1,1,1);
    Eigen::Vector3f dir(0,2,0);
    float d = openMVG::sfm::getDistToLine(pl,dir,pol);
    EXPECT_NEAR(0,d,threshold);
}

int main() { TestResult tr; return TestRegistry::runAllTests(tr);}
