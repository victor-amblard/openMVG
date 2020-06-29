#include "openMVG/sfm/sfm_line_utils.hpp"

#include "testing/testing.h"

//test getDistToLine
float threshold(0.0001f);
namespace openMVG{


TEST(DistanceFunctions, point2_line2){
    Vec2 pol(0,0);
    Vec2 pl(1,1);
    Vec2 dir(1,1);
    float d = openMVG::sfm::distPoint2Line2D(pl,dir,pol);
    EXPECT_NEAR(0,d,threshold);
}
TEST(DistanceFunctions, point2_segment2){
    Vec2 p1(0,0);
    Vec2 pl(0.5,0.5);
    Vec2 p2(1,1);
    openMVG::sfm::Endpoints2 ep = std::make_pair(p1, p2);
    Vec2 normal(1,1);
    float d = openMVG::sfm::distPoint2Segment(pl, ep, normal);
    EXPECT_NEAR(0,d,threshold);
}

int main() { TestResult tr; return TestRegistry::runAllTests(tr);}
}