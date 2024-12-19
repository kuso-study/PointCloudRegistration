import open3d as o3d
import numpy as np
import os
import glob
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
    file_paths = glob.glob(os.path.join(folder_path, file_extension))
    return file_paths

def execute_icp(target, source, init_transform, max_correspondence_distance, voxel_size):
    """
    执行 ICP 配准
    """
    treg = o3d.t.pipelines.registration
    estimation = treg.TransformationEstimationPointToPoint()
    criteria = treg.ICPConvergenceCriteria(relative_fitness=0.000001, relative_rmse=0.000001, max_iteration=6000)
    registration_icp = treg.icp(
        source, target, max_correspondence_distance, init_transform, estimation, criteria, voxel_size
    )
    return registration_icp

def visualize_combined_point_clouds(merged_pcd):
    """
    可视化所有点云文件配准后的整体结果。

    :param point_clouds: 所有配准后的点云列表
    """
    merged_pcd_temp = merged_pcd.to_legacy()
    o3d.visualization.draw_geometries([merged_pcd_temp])

def main():
    merged_pcd = o3d.t.geometry.PointCloud()
    voxel_size = 1

    # 获取点云文件夹中的所有点云文件路径
    folder_path = "D:/study/Ureca/pointcloud"
    point_cloud_paths = get_point_cloud_paths(folder_path)

    if len(point_cloud_paths) < 2:
        print("点云文件不足，无法进行配准。")
        return

    # 读取第一个点云作为基准
    base_pcd = o3d.t.io.read_point_cloud(point_cloud_paths[0])
    combined_pcds = [base_pcd]
    merged_pcd = base_pcd.clone()

    for i in range(4, 5):
        print(f"正在处理: {point_cloud_paths[i]}")

        # 读取当前点云
        source_pcd = o3d.t.io.read_point_cloud(point_cloud_paths[i])

        # 使用全局配准获取初始变换矩阵
        result_pre = hp.execute_global_registration(merged_pcd, source_pcd, voxel_size)
        print(f"第{i}个点云的RANSAC配准变换矩阵:\n", result_pre.transformation)

        # 使用 ICP 进行精细配准
        init_transform = result_pre.transformation
        
        max_correspondence_distance = 2 * voxel_size
        registration_icp = execute_icp(merged_pcd, source_pcd, init_transform, max_correspondence_distance, voxel_size)

        print(f"第{i}个点云的ICP配准变换矩阵:\n", registration_icp.transformation)

        # 更新 base_pcd 为当前配准后的点云
        source_pcd.transform(registration_icp.transformation)
        combined_pcds.append(source_pcd)
        merged_pcd = source_pcd + merged_pcd

    # 可视化所有点云配准后的整体结果
    visualize_combined_point_clouds(merged_pcd)

if __name__ == "__main__":
    main()
