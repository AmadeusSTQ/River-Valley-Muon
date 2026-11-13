import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle, ConnectionPatch

# ===== 1. 读取 CSV =====
csv_path = "muon1000val.csv"

df = pd.read_csv(csv_path)

# ===== 2. 设置横轴和曲线列 =====
x_col = "iter"
y_cols = [col for col in df.columns if col.endswith(" - val/loss")]

# ===== 3. 准备最后 501 步的数据 =====
max_iter = df[x_col].max()

# 这里是最后 501 步：比如 3500 到 4000，包含两个端点，所以是 501 个 step
zoom_x0 = max_iter - 500
zoom_x1 = max_iter

last500_df = df[(df[x_col] >= zoom_x0) & (df[x_col] <= zoom_x1)]

# ===== 4. 创建主图 =====
fig, ax = plt.subplots(figsize=(12, 4))

for col in y_cols:
    data = df[[x_col, col]].dropna()
    label = col.replace(" - val/loss", "")

    ax.plot(
        data[x_col],
        data[col],
        marker="o",
        markersize=3,
        linewidth=1.5,
        label=label
    )

ax.set_title("Muon1000+AdamW3000 Iter Validation Loss Curves", fontsize=16)
ax.set_xlabel("Iter", fontsize=14)
ax.set_ylabel("Val loss", fontsize=14)
ax.tick_params(axis="both", labelsize=12)
ax.grid(True, alpha=0.3)

# ===== 5. 图例放到大图内部，上半部分，横向排列 =====
ax.legend(
    fontsize=8,
    loc="upper center",
    bbox_to_anchor=(0.05, 0.9, 0.95, 0.08),
    ncol=len(y_cols),
    mode="expand",
    frameon=True,
    fancybox=True,
    framealpha=0.85,
    borderpad=0.35,
    columnspacing=0.6,
    handlelength=1.2,
    handletextpad=0.35,
    markerscale=0.9,
    borderaxespad=0.1,
    labelspacing=0.2
)

# 图例已经在图内，所以不用给右侧留空
fig.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.90)

# ===== 6. 创建 inset 小图 =====
axins = ax.inset_axes([0.40, 0.30, 0.50, 0.50])

for col in y_cols:
    data = last500_df[[x_col, col]].dropna()

    axins.plot(
        data[x_col],
        data[col],
        marker="o",
        markersize=2,
        linewidth=1.2
    )

axins.set_title("3500-4000 Iter", fontsize=11)
axins.set_xlabel("Iter", fontsize=9)
axins.set_ylabel("Val loss", fontsize=9)
axins.tick_params(axis="both", labelsize=8)
axins.grid(True, alpha=0.3)

# 小图横轴范围
axins.set_xlim(zoom_x0, zoom_x1)

# 小图 y 轴范围
y_min = last500_df[y_cols].min().min()
y_max = last500_df[y_cols].max().max()
margin = (y_max - y_min) * 0.08
zoom_y0 = y_min - margin
zoom_y1 = y_max + margin
axins.set_ylim(zoom_y0, zoom_y1)
rect_x0 = zoom_x0 - 30
rect_x1 = zoom_x1 + 30
rect_y0 = zoom_y0 - 0.02
rect_y1 = zoom_y1 + 0.02

# ===== 7. 给 inset 小图本身加粗黑色边框 =====
for spine in axins.spines.values():
    spine.set_edgecolor("black")
    spine.set_linewidth(1.3)

# ===== 8. 在主图上框出最后 501 步对应的区域 =====
rect = Rectangle(
    (rect_x0, rect_y0),
    rect_x1 - rect_x0,
    rect_y1 - rect_y0,
    fill=False,
    edgecolor="black",
    linewidth=1.3,
    linestyle="--",
    zorder=20
)
ax.add_patch(rect)

# ===== 9. 添加带箭头的指示线：从主图框选区域指向 inset 小图 =====
rect_center_x = (rect_x0 + rect_x1) / 2
rect_center_y = (rect_y0 + rect_y1) / 2
arrow = ConnectionPatch(
    xyA=(rect_center_x, rect_center_y),      # 箭头起点：主图中框选区域的左上角
    coordsA=ax.transData,
    xyB=(0.95, 0.05),           # 箭头终点：小图右下角附近
    coordsB=axins.transAxes,
    axesA=ax,
    axesB=axins,
    arrowstyle="->",
    mutation_scale=18,
    linewidth=1.3,
    color="black",
    shrinkA=3,
    shrinkB=3
)
fig.add_artist(arrow)

# ===== 10. 保存成 PDF =====
plt.savefig(
    "val_loss_val2_inside_val1_with_arrow.pdf",
    format="pdf",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("已生成：val_loss_val2_inside_val1_with_arrow.pdf")