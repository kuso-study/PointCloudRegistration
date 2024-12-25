import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# 加载点云
point_cloud = o3d.io.read_point_cloud("D:/study/Ureca/pointcloud/merged_pcd.ply")
point_cloud = point_cloud.voxel_down_sample(voxel_size=1.5)
point_cloud = point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)[0]

# 计算主轴方向 (PCA)
points = np.asarray(point_cloud.points)
cov_matrix = np.cov(points.T)
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# 主轴方向
principal_axis = eigenvectors[:, np.argmax(eigenvalues)]

# 投影到主轴方向，计算范围
projections = np.dot(points, principal_axis)
min_proj, max_proj = projections.min(), projections.max()

# 主平面方向
secondary_axes = eigenvectors[:, np.argsort(eigenvalues)[-2:]]  # 主平面方向（次主轴）

# 点云投影到主平面
projected_points = np.dot(points - points.mean(axis=0), secondary_axes)


# 切片函数
def get_slice_points(slice_center, slice_thickness):
    """
    获取切片范围内的点。
    :param slice_center: 切片中心位置
    :param slice_thickness: 切片厚度
    :return: 切片内的点 (N, 3)
    """
    mask = (projections >= slice_center - slice_thickness / 2) & (projections < slice_center + slice_thickness / 2)
    return points[mask]


# 初始化切片参数
slice_thickness = 0.5
initial_center = (min_proj + max_proj) / 2
initial_slice = get_slice_points(initial_center, slice_thickness)

# 创建双子图窗口
fig, (ax_main, ax_slice) = plt.subplots(1, 2, figsize=(12, 6))
plt.subplots_adjust(bottom=0.25)

# 主平面投影绘制
ax_main.set_title("Model Projection on Principal Plane (Side View)")
ax_main.set_xlabel("Secondary Axis 1")
ax_main.set_ylabel("Secondary Axis 2")
ax_main.scatter(projected_points[:, 0], projected_points[:, 1], s=1, c="gray", label="Model Projection")

# 初始化红线，表示切片位置
slice_line, = ax_main.plot(
    [projected_points[:, 0].min(), projected_points[:, 0].max()],  # 横跨整个 Secondary Axis 1 范围
    [0, 0],  # 初始位置 y = 0
    'r-', lw=2, label="Slice Position"
)
ax_main.legend()

# 初始化切片展示
ax_slice.set_title("Real-Time Slice")
ax_slice.set_xlabel("X")
ax_slice.set_ylabel("Y")
scatter_slice = ax_slice.scatter(initial_slice[:, 0], initial_slice[:, 1], s=1, c="blue")


# 添加滑块
ax_slider = plt.axes([0.25, 0.1, 0.5, 0.03])  # 滑块位置
slider = Slider(ax_slider, 'Slice Center', min_proj, max_proj, valinit=initial_center)


# 更新函数
def update(val):
    slice_center = slider.val
    slice_points = get_slice_points(slice_center, slice_thickness)

    # 获取左侧图 Y 轴的上下界差值
    y_min, y_max = projected_points[:, 1].min(), projected_points[:, 1].max()
    y_range = y_max - y_min

    # 计算红线的 Y 坐标位置，考虑滑块值的偏移量
    slice_show = y_min + (slice_center - min_proj) * y_range / (max_proj - min_proj)

    # 更新主平面投影中的红线位置
    slice_line.set_data(
        [projected_points[:, 0].min(), projected_points[:, 0].max()],  # 红线 X 坐标横跨整个范围
        [slice_show, slice_show]  # 红线 Y 坐标
    )


    # 更新切片图
    if slice_points.size > 0:
        scatter_slice.set_offsets(slice_points[:, :2])  # 使用 X 和 Y 坐标
        ax_slice.set_xlim(slice_points[:, 0].min() - 1, slice_points[:, 0].max() + 1)
        ax_slice.set_ylim(slice_points[:, 1].min() - 1, slice_points[:, 1].max() + 1)
    else:
        scatter_slice.set_offsets([])  # 如果切片无点，清空右侧显示
    plt.draw()


# 绑定更新函数到滑块
slider.on_changed(update)

# 显示窗口
plt.show()
