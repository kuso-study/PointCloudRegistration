import open3d as o3d
treg = o3d.t.pipelines.registration
import numpy as np
import os
import glob
import time
import HeadPreprocess as hp

def draw_registration_result(source, target, transformation):
    # Clone to avoid modifying original data
    source_temp = source.clone()
    target_temp = target.clone()

    # Apply the transformation
    source_temp.transform(transformation)

    # Convert Tensor to Legacy for visualization
    source_temp = source_temp.to_legacy()
    target_temp = target_temp.to_legacy()

    # Visualize the transformed point clouds
    o3d.visualization.draw_geometries([source_temp, target_temp])

def get_point_cloud_paths(folder_path, file_extension="*.ply"):
    """
    获取指定文件夹中所有点云文件的路径列表。

    :param folder_path: 点云文件所在的文件夹路径
    :param file_extension: 文件扩展名，默认为 "*.ply"
    :return: 包含点云文件路径的列表
    """
    # 使用 glob 获取指定扩展名的文件
    file_paths = glob.glob(os.path.join(folder_path, file_extension))
    return file_paths

def execute_icp(source, target, init_transform, max_correspondence_distance, voxel_size):
    """
    执行 ICP 配准
    """
    estimation = treg.TransformationEstimationPointToPoint()
    criteria = treg.ICPConvergenceCriteria(relative_fitness=0.000001, relative_rmse=0.000001, max_iteration=50)
    registration_icp = treg.icp(
        source, target, max_correspondence_distance, init_transform, estimation, criteria, voxel_size
    )
    return registration_icp

def main(): # Small test on combining 0 & 1
    head_icp_pcds = get_point_cloud_paths("D:/study/Ureca/pointcloud")
    source = o3d.t.io.read_point_cloud(head_icp_pcds[0])
    target = o3d.t.io.read_point_cloud(head_icp_pcds[1])

    result_pre = hp.execute_global_registration(source, target, 0.8)
    print("RANSAC 配准结果的变换矩阵:", result_pre.transformation)
    
    init_transform = result_pre.transformation
    max_correspondence_distance = 0.07
    voxel_size = 0.025

    registration_icp = execute_icp(source, target, init_transform, max_correspondence_distance, voxel_size)
    print("ICP 配准结果的变换矩阵:", registration_icp.transformation)
    draw_registration_result(source, target, registration_icp.transformation)


if __name__ == "__main__":
    main()