# Google Colab 使用说明

## 快速开始

### 1. 安装依赖

在 Colab 的第一个代码单元格中运行：

```python
!pip install numpy pandas matplotlib
```

或者如果上传了 `requirements.txt`：

```python
!pip install -r requirements.txt
```

### 2. 上传数据文件

需要上传以下文件到 Colab：
- `partB_sessions.csv`
- `Experiment_B_Trading_Data.csv`

可以使用以下代码上传：

```python
from google.colab import files
uploaded = files.upload()
```

### 3. 运行脚本

将 `generate_two_figures.py` 的内容复制到 Colab 单元格中运行，或者：

```python
# 如果上传了文件
exec(open('generate_two_figures.py').read())
```

### 4. 查看生成的图片

脚本会生成两个图片文件：
- `figs/mc_null_distribution.png` - Monte Carlo 零分布图
- `choiceB_scatter_absolute.png` - 散点图

在 Colab 中查看图片：

```python
from IPython.display import Image, display

# 显示图片
display(Image('choiceB_scatter_absolute.png'))
display(Image('figs/mc_null_distribution.png'))
```

## 依赖包

- numpy >= 1.20.0
- pandas >= 1.3.0
- matplotlib >= 3.3.0

## 注意事项

- Colab 已经预装了大部分常用库，但建议运行 `!pip install` 确保版本兼容
- 在 Colab 中，`matplotlib.use('Agg')` 会自动处理，不影响使用
- 确保数据文件路径正确

