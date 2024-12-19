import open3d as o3d
import numpy as np
import glob
import os

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



def main():
    # 设置体素大小，用于降采样和特征提取
    voxel_size = 0.5

    # 加载源点云和目标点云
    head_icp_pcds = get_point_cloud_paths("D:/study/Ureca/pointcloud")
    source = o3d.io.read_point_cloud(head_icp_pcds[0])
    target = o3d.io.read_point_cloud(head_icp_pcds[1])

    # print("原始源点云点数:", len(source.points))
    # print("原始目标点云点数:", len(target.points))

    # 预处理：降采样、法线估计、FPFH 特征提取
    # source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    # target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

    # print("降采样后的源点云点数:", len(source_down.points))
    # print("降采样后的目标点云点数:", len(target_down.points))
    # print("源点云 FPFH 特征维度:", source_fpfh.data.shape)
    # print("目标点云 FPFH 特征维度:", target_fpfh.data.shape)

    # 执行全局配准
    result_ransac = execute_global_registration(source, target, voxel_size)
    print("RANSAC 配准结果的变换矩阵:")
    print(result_ransac.transformation)

    # 将变换应用到源点云
    #source_down.transform(result_ransac.transformation)

    # 可视化初步配准结果
    # print("显示初步配准结果...")
    # o3d.visualization.draw_geometries([source_down, target_down])

if __name__ == "__main__":
    main()

