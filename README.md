# OCR
OCR demo

#### 
```bash
# install the latest version of tesseract for MacOS
brew install tesseract --head

# install pytesseract via pip
pip install pytesseract

# download language models from https://github.com/tesseract-ocr/tessdata
# Then move these models to /usr/local/share/tessdata/
chi_sim.traineddata (Simplified Chinese)
chi_tra.traineddata (Traditional Chinese)
```

```markdown
OCR language:　识别图像中字体中的语言，在命令行和pytesseract，使用-l 选项
OCR Engine Mode(oem):tesseract4有2个ocr引擎(legacy,lstm),用—oem选项去设置
0 Legacy engine only.
1 Neural nets LSTM engine only.
2 Legacy + LSTM engines.
3 Default, based on what is available.
Page Segmentation Mode(psm): psm 或许是非常有用的，对于结构化文本有额外的信息对于python和命令行工具默认是3.
0 只有方向和脚本检测（OSD）。
1 使用OSD自动分页。
2 自动分页，但没有OSD或OCR。
3 全自动页面分割，但没有OSD。（默认）
4 假设一列可变大小的文本。
5 假设一个统一的垂直排列文本块。
6 假设一个统一的文本块。
7 将图像作为单个文本行处理。
8 将图像视为一个单词。
9 将图像视为一个圆圈中的单个单词。
10 将图像视为单个字符。
```
