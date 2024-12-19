import open3d as o3d
import numpy as np
import glob
import os

def load_point_clouds(folder_path, voxel_size=0.05):
    file_paths = glob.glob(os.path.join(folder_path, "*.ply"))
    point_clouds = []
    for path in file_paths:
        pcd = o3d.io.read_point_cloud(path)
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        point_clouds.append(pcd)
    return point_clouds

def pairwise_registration(source, target):
    source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    result_icp = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance=0.05,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
    )
    return result_icp.transformation

def full_registration(point_clouds, max_correspondence_distance=0.05):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))

    for i in range(len(point_clouds) - 1):
        source = point_clouds[i]
        target = point_clouds[i + 1]
        transformation_icp = pairwise_registration(source, target)
        odometry = np.dot(transformation_icp, odometry)
        pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry)))
        pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(i, i + 1, transformation_icp, uncertain=False))
    
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=max_correspondence_distance,
        edge_prune_threshold=0.25,
        reference_node=0)
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        method=o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        criteria=o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option=option)
    
    return pose_graph

def transform_point_clouds(point_clouds, pose_graph):
    for i, pcd in enumerate(point_clouds):
        pcd.transform(pose_graph.nodes[i].pose)
    return point_clouds

def main():
    folder_path = "D:/study/Ureca/pointcloud"  # 修改为你的文件夹路径
    point_clouds = load_point_clouds(folder_path)
    pose_graph = full_registration(point_clouds)
    aligned_point_clouds = transform_point_clouds(point_clouds, pose_graph)

    # 合并所有点云并显示
    combined = o3d.geometry.PointCloud()
    for pcd in aligned_point_clouds:
        combined += pcd
    o3d.visualization.draw_geometries([combined])

if __name__ == "__main__":
    main()
