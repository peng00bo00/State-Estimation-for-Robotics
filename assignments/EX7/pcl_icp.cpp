#include <iostream>
#include <string>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/transforms.h>
// #include <pcl/console/time.h>   // TicToc

#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

string first  = "./first.pcd";
string second = "./second.pcd";

void load_pcd(PointCloudT::Ptr &cloud, string &filepath) {
    if (pcl::io::loadPCDFile<PointT> (filepath, *cloud) == -1)
    {
        PCL_ERROR ("Couldn't read pcd file. \n");
    }
}

void visualize_pcd(PointCloudT::Ptr &cloud1, PointCloudT::Ptr &cloud2) {
    pcl::visualization::PCLVisualizer viewer ("ICP");

    // first point cloud is green
    pcl::visualization::PointCloudColorHandlerCustom<PointT> green (cloud1, 20, 180, 20);
    viewer.addPointCloud (cloud1, green, "cloud1");

    // second point cloud is red
    pcl::visualization::PointCloudColorHandlerCustom<PointT> red (cloud2, 180, 20, 20);
    viewer.addPointCloud (cloud2, red, "cloud2");

    viewer.setBackgroundColor (0.0, 0.0, 0.0);

    while (!viewer.wasStopped ())
    {
        viewer.spinOnce ();
    }
}

int main() {
    // load the 2 point clouds
    PointCloudT::Ptr cloud1 (new PointCloudT);
    PointCloudT::Ptr cloud2 (new PointCloudT);

    load_pcd(cloud1, first);
    load_pcd(cloud2, second);

    cout << "Successfully load point clouds!" << endl;

    // visualize before registeration
    // visualize_pcd(cloud1, cloud2);
    cout << "Start registeration..." << endl;

    // Set the input source and target
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setInputCloud (cloud1);
    icp.setInputTarget (cloud2);

    pcl::PointCloud<pcl::PointXYZ> Final;

    // Perform the alignment
    icp.align (Final);
    // Obtain the transformation that aligned cloud_source to cloud_source_registered
    Eigen::Matrix4f transformation = icp.getFinalTransformation ();
    cout << "T = " << endl << transformation << endl;
    cout << "Pose = " << endl << transformation.inverse() << endl;
    
    PointCloudT::Ptr transformed_cloud1( new PointCloudT());
    pcl::transformPointCloud(*cloud1, *transformed_cloud1, transformation);
    visualize_pcd(transformed_cloud1, cloud2);

    Matrix3f C = Matrix3f::Identity();
    Vector3f r = Vector3f::Zero();

    return 0;
}