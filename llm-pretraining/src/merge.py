import fitz  # PyMuPDF


# ===== 1. 输入 PDF 文件 =====
top_pdf = "v muon1000val.pdf"      # 放在上面的图
bottom_pdf = "v muon1500val.pdf"   # 放在下面的图

output_pdf = "merged_vertical.pdf"


# ===== 2. 自动获取页面真实内容区域，尽量去掉白边 =====
def get_content_rect(page):
    """
    获取 PDF 页面中实际有内容的区域。
    这个函数会尽量根据文字、线条、图形等元素计算边界。
    """
    rects = []

    # 文字区域
    for block in page.get_text("blocks"):
        x0, y0, x1, y1 = block[:4]
        rects.append(fitz.Rect(x0, y0, x1, y1))

    # 图形/线条区域，比如 matplotlib 画出来的坐标轴、曲线、边框
    for drawing in page.get_drawings():
        if "rect" in drawing:
            rects.append(drawing["rect"])

    if not rects:
        return page.rect

    content_rect = rects[0]
    for r in rects[1:]:
        content_rect |= r

    # 轻微扩展一点点，避免线条或文字被裁掉
    padding = 1
    content_rect.x0 -= padding
    content_rect.y0 -= padding
    content_rect.x1 += padding
    content_rect.y1 += padding

    # 防止超出原页面
    content_rect &= page.rect

    return content_rect


# ===== 3. 打开两个 PDF =====
doc_top = fitz.open(top_pdf)
doc_bottom = fitz.open(bottom_pdf)

page_top = doc_top[0]
page_bottom = doc_bottom[0]

rect_top = get_content_rect(page_top)
rect_bottom = get_content_rect(page_bottom)

w_top = rect_top.width
h_top = rect_top.height

w_bottom = rect_bottom.width
h_bottom = rect_bottom.height


# ===== 4. 创建新 PDF 页面 =====
out = fitz.open()

new_width = max(w_top, w_bottom)
new_height = h_top + h_bottom

new_page = out.new_page(width=new_width, height=new_height)


# ===== 5. 让两个图水平居中 =====
top_x0 = (new_width - w_top) / 2
top_x1 = top_x0 + w_top

bottom_x0 = (new_width - w_bottom) / 2
bottom_x1 = bottom_x0 + w_bottom


# ===== 6. 放置上面的图 =====
new_page.show_pdf_page(
    fitz.Rect(top_x0, 0, top_x1, h_top),
    doc_top,
    0,
    clip=rect_top
)


# ===== 7. 放置下面的图，紧贴上图 =====
new_page.show_pdf_page(
    fitz.Rect(bottom_x0, h_top, bottom_x1, h_top + h_bottom),
    doc_bottom,
    0,
    clip=rect_bottom
)


# ===== 8. 保存 =====
out.save(output_pdf)
out.close()
doc_top.close()
doc_bottom.close()

print(f"已生成：{output_pdf}")