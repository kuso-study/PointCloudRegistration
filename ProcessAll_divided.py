import open3d as o3d
import numpy as np
import os
import glob

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

def preprocess_point_cloud(pcd, voxel_size):
    """
    对点云进行降采样，并计算法线和 FPFH 特征
    :param pcd: 输入的点云
    :param voxel_size: 体素大小，用于降采样
    :return: 降采样后的点云和对应的 FPFH 特征
    """
    # 检查点云类型
    # if not isinstance(pcd, o3d.geometry.PointCloud):
    #     pcd = o3d.geometry.PointCloud(pcd.points)
    if isinstance( pcd, o3d.t.geometry.PointCloud):
        pcd =  pcd.to_legacy()

    # 降采样
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down, ind = pcd_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    print(f"降采样后的点云点数: {len(pcd_down.points)}")

    # 估计法线（经典 API）
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )

    # 计算 FPFH 特征
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100)
    )
    return pcd_down, fpfh

def execute_global_registration(target, source, voxel_size):
    """
    执行基于 FPFH 特征的 RANSAC 全局配准
    """
    # source.translate(-np.mean(np.asarray(source.points), axis=0))
    # target.translate(-np.mean(np.asarray(target.points), axis=0))

    # 预处理点云
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

    # 设置 RANSAC 配准参数
    distance_threshold = voxel_size * 1  # RANSAC 最大对应点距离
    print("RANSAC 配准中，距离阈值: %.3f" % distance_threshold)

    # 执行 RANSAC
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        mutual_filter=False,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
    )
    return result

def execute_icp(target, source, init_transform, max_correspondence_distance, voxel_size):
    """
    执行 ICP 配准
    """
    treg = o3d.t.pipelines.registration
    estimation = treg.TransformationEstimationPointToPoint()
    criteria = treg.ICPConvergenceCriteria(relative_fitness=0.000001, relative_rmse=0.000001, max_iteration=1000)
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
    merged_pcd = []
    final_pcd = o3d.t.geometry.PointCloud()
    cnt = 0
    manual_init = [np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
                   np.asarray([[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]]),
                   np.asarray([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]),
                   np.asarray([[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])]

    # 获取点云文件夹中的所有点云文件路径
    folder_path = "D:/study/Ureca/pointcloud"
    point_cloud_paths = get_point_cloud_paths(folder_path)
    
    point_clouds = [o3d.t.io.read_point_cloud(path) for path in point_cloud_paths]

    group_size = len(point_clouds) // 4
    groups = [point_clouds[i:i + group_size] for i in range(0, len(point_clouds), group_size)]
    voxel_size = 1.5

    for group in groups:
        base_pcd = group[0]
        merged_pcd.append(base_pcd)

        for i in range(1, len(group)):
            print(f"正在处理: {point_cloud_paths[i]}")

            source_pcd = group[i]

            result_pre = execute_global_registration(merged_pcd[cnt], source_pcd, 1.5)
            print(f"第{i}个点云的RANSAC配准变换矩阵:\n", result_pre.transformation)

            init_transform = result_pre.transformation
            max_correspondence_distance = 1.5 * voxel_size
            registration_icp = execute_icp(merged_pcd[cnt], source_pcd, init_transform, max_correspondence_distance, voxel_size)

            print(f"第{i}个点云的ICP配准变换矩阵:\n", registration_icp.transformation)

            source_pcd.transform(registration_icp.transformation)
            merged_pcd[cnt] = source_pcd + merged_pcd[cnt]

        merged_pcd[cnt].transform(manual_init[cnt])
        #visualize_combined_point_clouds(merged_pcd[cnt])
        cnt += 1
        
    final_pcd = merged_pcd[0]
    for i in range(1, len(merged_pcd)):
        print(f"正在处理最终合并")
        source_pcd = merged_pcd[i]

        result_pre = execute_global_registration(final_pcd, merged_pcd[i], 1)
        print(f"第{i}个点云的RANSAC配准变换矩阵:\n", result_pre.transformation)

        init_transform = result_pre.transformation
        max_correspondence_distance = 1.5 * voxel_size
        registration_icp = execute_icp(final_pcd, merged_pcd[i], init_transform, max_correspondence_distance, voxel_size)

        print(f"第{i}个点云的ICP配准变换矩阵:\n", registration_icp.transformation)

        source_pcd.transform(registration_icp.transformation)
        final_pcd = source_pcd + final_pcd
    visualize_combined_point_clouds(final_pcd)

    # Output the final point cloud
    #final_pcd.points = o3d.utility.Vector3dVector(np.asarray(final_pcd.point).astype(np.float32))
    final_pcd = final_pcd.to_legacy()
    o3d.io.write_point_cloud("D:/study/Ureca/pointcloud/merged_pcd.ply", final_pcd)

if __name__ == "__main__":
    main()