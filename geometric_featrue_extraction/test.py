import cadquery as cq
import cadquery as cq
import numpy as np

# 加载 STEP 文件
part = cq.importers.importStep("my_part.stp")

# 获取几何对象
solid = part.val()

# 特征提取
bbox = solid.BoundingBox()
volume = solid.Volume()
surface_area = solid.Area()
center = solid.Center()

# 统计结构复杂度
num_faces = len(solid.Faces())
num_edges = len(solid.Edges())
num_solids = len(part.solids().vals())

# 构建向量
feature_vector = [
    bbox.xlen, bbox.ylen, bbox.zlen,       # 尺寸
    volume, surface_area,                  # 体积+表面积
    center.x, center.y, center.z,          # 重心
    num_faces, num_edges, num_solids       # 拓扑结构
]

print("几何特征向量:", feature_vector)

import numpy as np
import trimesh

# STEP 文件导入
shape = cq.importers.importStep("my_part.stp").val()

# 网格化（生成顶点 + 面片）
raw_vertices, raw_faces = shape.tessellate(1.0)

# 将 Vector 类型转为标准 float 数组
vertices = np.array([[v.x, v.y, v.z] for v in raw_vertices], dtype=np.float64)
faces = np.array(raw_faces, dtype=np.int64)

# 构建 trimesh 对象
mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

# 可视化（可选）
mesh.show()
