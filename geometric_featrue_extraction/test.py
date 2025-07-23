import cadquery as cq

# 读取 step 文件
part = cq.importers.importStep("my_part.stp")

# 提取体积、质心、包围盒
solid = part.val()
volume = solid.Volume()
area = solid.Area()
center = solid.Center()
bbox = solid.BoundingBox()

features = [volume, area, bbox.xlen, bbox.ylen, bbox.zlen]
print("几何特征向量:", features)
