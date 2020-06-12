#include <iostream>
#include <string>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/transforms.h>

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

void ICP_step(PointCloudT::Ptr &p, PointCloudT::Ptr &y, Matrix3f &C, Vector3f &r, float &e) {
    // set up a kd-tree
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(y);

    std::vector<int> pointIdxNKNSearch(0);
    std::vector<float> pointNKNSquaredDistance(0);
    e = 0;

    // find the closest point in y for every point in p
    PointCloudT::Ptr closest( new PointCloudT());

    for (size_t i = 0; i < p->points.size (); i++)
    {        
        kdtree.nearestKSearch(p->points[i], 1, pointIdxNKNSearch, pointNKNSquaredDistance);
        closest->push_back(y->points[pointIdxNKNSearch[0]]);
        e += pointNKNSquaredDistance[0];
    }

    // get centroids
    Eigen::Vector4f pp, yy;  
    pcl::compute3DCentroid(*p,       pp);
    pcl::compute3DCentroid(*closest, yy);

    // move to centroids
    Matrix4f T_p = Matrix4f::Identity();
    Matrix4f T_y = Matrix4f::Identity();

    T_p.col(3) = -pp;
    T_y.col(3) = -yy;

    PointCloudT::Ptr q( new PointCloudT());
    PointCloudT::Ptr qq(new PointCloudT());
    pcl::transformPointCloud(*p,        *q, T_p);
    pcl::transformPointCloud(*closest, *qq, T_y);

    // get W matrix
    Matrix3f W = Matrix3f::Zero();
    for (size_t i = 0; i < p->points.size(); i++)
    {
        W += qq->points[i].getVector3fMap() * q->points[i].getVector3fMap().transpose() / p->points.size();
    }

    // SVD
    JacobiSVD<Eigen::MatrixXf> svd(W, ComputeThinU | ComputeThinV );
    Matrix3f V = svd.matrixV(), U = svd.matrixU();
    DiagonalMatrix<float, 3> S(1, 1, U.determinant()*V.determinant());

    C = U * S * V.transpose();
    r = pp.head<3>() - C.transpose() * yy.head<3>();
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

    // ICP
    Matrix3f C = Matrix3f::Identity();
    Vector3f r = Vector3f::Zero();

    float e = 0;
    int i = 0;

    ICP_step(cloud1, cloud2, C, r, e);

    cout << "Iteration " << i << ", " << "Total error = " << e << endl;

    Matrix4f T = Matrix4f::Identity();
    T.block(0,0,3,3) = C;
    T.block(0,3,3,1) = -C*r;

    PointCloudT::Ptr transformed_cloud1( new PointCloudT());
    pcl::transformPointCloud(*cloud1, *transformed_cloud1, T);
    
    float ee = e;

    while (ee <= e)
    {
        i += 1;
        e = ee;
        pcl::transformPointCloud(*cloud1, *transformed_cloud1, T);
        ICP_step(transformed_cloud1, cloud2, C, r, ee);
        cout << "Iteration " << i << ", " << "Total error = " << ee << endl;

        Matrix4f dT = Matrix4f::Identity();
        dT.block(0,0,3,3) = C;
        dT.block(0,3,3,1) = -C*r;

        T = dT * T;
    }

    // visualize after registeration
    pcl::transformPointCloud(*cloud1, *transformed_cloud1, T);
    
    cout << "T = " << endl << T << endl;
    cout << "Pose = " << endl << T.inverse() << endl;

    visualize_pcd(transformed_cloud1, cloud2);

    return 0;
}
