import json
from collections import defaultdict
from playwright.sync_api import sync_playwright

def extract_visual_attributes(svg_path):
    # 使用 Playwright 进行无头浏览器操作
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)  # 启动无头浏览器
        page = browser.new_page()

        # 读取 SVG 文件内容并直接加载
        with open(svg_path, 'r', encoding='utf-8') as svg_file:
            svg_content = svg_file.read()

        # 加载 SVG 内容到页面中
        page.set_content(svg_content)

        # 等待 SVG 加载完成
        page.wait_for_selector('svg')

        # JavaScript 脚本：获取视觉相关的属性，包括位置信息和尺寸
        js_script = """
        () => {
            const svg = document.querySelector('svg');
            const elements_info = [];
            if (svg) {
                const elements = svg.querySelectorAll('*'); // 获取所有 SVG 内元素
                elements.forEach(element => {
                    const tagName = element.tagName;
                    const visualAttributes = {};

                    // 提取所有属性，包括位置信息和尺寸
                    const attrs = element.attributes;
                    for (let i = 0; i < attrs.length; i++) {
                        const attrName = attrs[i].name;
                        const attrValue = attrs[i].value;
                        visualAttributes[attrName] = attrValue;
                    }

                    // 提取通用的样式属性
                    const computedStyle = window.getComputedStyle(element);
                    ['fill', 'stroke', 'stroke-width', 'opacity', 'fill-opacity', 'stroke-opacity'].forEach(styleProp => {
                        const value = computedStyle.getPropertyValue(styleProp);
                        if (value && value !== 'none' && value !== '0') {
                            visualAttributes[styleProp] = value;
                        }
                    });

                    // 如果 visualAttributes 非空，则添加到列表
                    if (Object.keys(visualAttributes).length > 0) {
                        elements_info.push({
                            'tagName': tagName,
                            'attributes': visualAttributes
                        });
                    }
                });
            }
            return elements_info;
        }
        """

        # 执行 JS 脚本并获取结果
        elements_info = page.evaluate(js_script)

        # 关闭浏览器
        browser.close()

        # 生成带有序号的元素名称
        tag_count = defaultdict(int)
        for item in elements_info:
            tag_name = item['tagName']
            tag_count[tag_name] += 1
            item['element'] = f"{tag_name}_{tag_count[tag_name]}"  # 按顺序为元素加上序号
            del item['tagName']  # 删除不再需要的原始标签名

        # 返回结果
        return elements_info

# 调用函数，传入 SVG 文件的路径
svg_file_path = "6.svg"
visual_attributes = extract_visual_attributes(svg_file_path)

# 输出最终的视觉属性
print(json.dumps(visual_attributes, indent=4, ensure_ascii=False))
