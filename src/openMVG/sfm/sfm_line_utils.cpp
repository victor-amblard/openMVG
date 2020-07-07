#include "openMVG/sfm/sfm_line_utils.hpp"
namespace openMVG {
namespace sfm {

double getCoeffLine(const Vec2& poi,
                    const Vec2& normal,
                    const Vec2& refPoint)
{
    return (poi-refPoint).dot(normal);
}

double pointToLineDist(const Vec2& point,
                       const Endpoints2& line)
{
    const double num = std::fabs( (line.second(1)-line.first(1))*point(0)-(line.second(0)-line.first(0))*point(1)+line.second(0)*line.first(1)-line.second(1)*line.first(0));
    const double den = pow(pow(line.second(1)-line.first(1),2)+pow(line.second(0)-line.first(0),2), 1/2.);

    return num/den;
}

void projectPoint2Line2D(const Vec2& curPoint,
                         const Vec2& pointLine,
                         const Vec2& normalizedDirLine,
                         Vec2& nPoint)
{
    Vec2 vec2Point = curPoint - pointLine;
    nPoint = pointLine + normalizedDirLine.dot(vec2Point)/normalizedDirLine.squaredNorm()*normalizedDirLine;
}

double distPoint2Line2D(const Vec2& curPoint,
                        const Vec2& pointLine,
                        const Vec2& dirLine)
{
    Vec2 projPoint;
    projectPoint2Line2D(curPoint, pointLine, dirLine, projPoint);

    return (projPoint-curPoint).norm();
}
double distPoint2Segment(const Vec2& point,
                         const Endpoints2& endpoints,
                         const Vec2 normal)
{
    Vec2 vec2Point = point - endpoints.first;
    double tEnd = normal.dot(endpoints.second - endpoints.first)/normal.squaredNorm();
    double coeff = normal.dot(vec2Point)/normal.squaredNorm();
    coeff =  std::max(0., std::min(tEnd, coeff));

    Vec2 projPoint = endpoints.first + coeff * normal;
    return (projPoint-point).norm();


}

Vec2 equation2Normal2D(const Vec3& eqn)
{
    return Vec2(-eqn(1), eqn(0)).normalized();
}
double CLIP_ANGLE(double angle){
    if (angle > M_PI)
        angle -= M_PI;
    if (angle < M_PI && angle > M_PI/2)
        angle = M_PI - angle;
    
    return angle;
}
Vec2 endpoints2Normal(Endpoints2 endpoints){
    return Vec2(endpoints.second(0)-endpoints.first(0), endpoints.second(1)-endpoints.first(1)).normalized();
}
double angleBetweenLines(const Endpoints2& lineA,
                         const Endpoints2& lineB)
{
    Vec2 normal1 = endpoints2Normal(lineA);
    Vec2 normal2 = endpoints2Normal(lineB);

    double angle = CLIP_ANGLE(std::acos(normal1.dot(normal2)));

    return angle;
}

double getRange(const ::pcl::PointXYZ& p)
{
    return pow(pow(p.x,2.)+pow(p.y,2.)+pow(p.z,2.), 1/2.);
}
bool segmentComp(std::pair<int, Segment3D>& p1, std::pair<int, Segment3D>& p2)
{
    if (p1.second < p2.second)
        return false;
    else
        return true;
}

int root(int i, std::vector<int>& clustersSegment){
    int root = i;

    while (root != clustersSegment.at(i)){
        root = clustersSegment.at(i);
    }
    //Path compression
    while (root != i){
        int nP = clustersSegment.at(i);
        clustersSegment.at(i) = root;
        i = nP;
    }
    return root;
}
void joinSets(int i1, int i2,  std::vector<int>& clustersSegment, int * rank)
{
    int x1 = root(i1, clustersSegment);
    int x2 = root(i2, clustersSegment);

    if (x1!=x2){
        if (rank[x1] < rank[x2]){
            rank[x1] += rank[x2];
            clustersSegment.at(x2) = x1;
        }else{
            rank[x2] += rank[x1];
            clustersSegment.at(x1) = x2;
        }
    }
}

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

Vec3 projectC2I(const Vec3& pt, const Mat3& K){
    return Vec3(K(0,2)+K(0,0)*pt(0)/pt(2), K(1,2)+K(1,1)*pt(1)/pt(2), pt(2));
}
Vec3 projectW2I(const Vec3& pt, const Mat34& projMatrix){
    return projMatrix * pt.homogeneous();
}
void ShowManyImages(std::string title, int nArgs, std::vector<cv::Mat>& imgs) {
int size;
int i;
int m, n;
int x, y;

// w - Maximum number of images in a row
// h - Maximum number of images in a column
int w, h;

// scale - How much we have to resize the image
float scale;
int max;

// If the number of arguments is lesser than 0 or greater than 12
// return without displaying
if(nArgs <= 0) {
    printf("Number of arguments too small....\n");
    return;
}
else if(nArgs > 14) {
    printf("Number of arguments too large, can only handle maximally 12 images at a time ...\n");
    return;
}
// Determine the size of the image,
// and the number of rows/cols
// from number of arguments
else if (nArgs == 1) {
    w = h = 1;
    size = 300;
}
else if (nArgs == 2) {
    w = 2; h = 1;
    size = 300;
}
else if (nArgs == 3 || nArgs == 4) {
    w = 2; h = 2;
    size = 300;
}
else if (nArgs == 5 || nArgs == 6) {
    w = 3; h = 2;
    size = 200;
}
else if (nArgs == 7 || nArgs == 8) {
    w = 4; h = 2;
    size = 200;
}
else {
    w = 4; h = 3;
    size = 200;
}

// Create a new 3 channel image
cv::Mat DispImage = cv::Mat::zeros(cv::Size(100 + size*w, 60 + size*h), CV_8UC3);

// Used to get the arguments passed


// Loop for nArgs number of arguments
for (i = 0, m = 20, n = 20; i < nArgs; i++, m += (20 + size)) {
    // Get the Pointer to the IplImage
    cv::Mat img = imgs.at(i);

    // Check whether it is NULL or not
    // If it is NULL, release the image, and return
    if(img.empty()) {
        printf("Invalid arguments");
        return;
    }

    // Find the width and height of the image
    x = img.cols;
    y = img.rows;

    // Find whether height or width is greater in order to resize the image
    max = (x > y)? x: y;

    // Find the scaling factor to resize the image
    scale = (float) ( (float) max / size );

    // Used to Align the images
    if( i % w == 0 && m!= 20) {
        m = 20;
        n+= 20 + size;
    }

    // Set the image ROI to display the current image
    // Resize the input image and copy the it to the Single Big Image
    cv::Rect ROI(m, n, (int)( x/scale ), (int)( y/scale ));
    cv::Mat temp; resize(img,temp, cv::Size(ROI.width, ROI.height));
    temp.copyTo(DispImage(ROI));
}

// Create a new window, and show the Single Big Image
cv::namedWindow( title, 1 );
cv::imshow( title, DispImage);
cv::waitKey();

// End the number of arguments
}
} //namespace sfm
} //namespace openMVG