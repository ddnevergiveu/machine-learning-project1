import matplotlib
matplotlib.use('Agg')  # 非交互式后端，只用于生成图片，不显示窗口
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager as fm
import os

# ---------------- 中文字体 ----------------
def get_chinese_font():
    candidate_fonts = [
        "/System/Library/AssetsV2/com_apple_MobileAsset_Font7/3419f2a427639ad8c8e139149a287865a90fa17e.asset/AssetData/PingFang.ttc",
        "/System/Library/AssetsV2/com_apple_MobileAsset_Font7/62032b9b64a0e3a9121c50aeb2ed794e3e2c201f.asset/AssetData/Hei.ttf",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
    ]
    for path in candidate_fonts:
        if os.path.exists(path):
            return fm.FontProperties(fname=path)
    raise RuntimeError("未找到可用中文字体")

my_font = get_chinese_font()

# ---------------- 可视化函数 ----------------
def plot_feature_importance(model, X, top_n=10, save_path=None, show=True):
    """
    绘制模型特征重要性
    :param model: 模型对象，需有 feature_importances_ 或 coef_ 属性
    :param X: 特征数据集
    :param top_n: 展示前 n 个重要特征
    :param save_path: 可选，保存图片路径
    :param show: 是否显示图片
    """
    # 获取特征重要性
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = model.coef_[0]
    else:
        raise ValueError("模型没有 feature_importances_ 或 coef_ 属性")

    feature_names = X.columns
    imp_df = sorted(zip(feature_names, importances), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    features, values = zip(*imp_df)

    plt.figure(figsize=(10, 6))
    # 解决 palette 警告，不使用 hue
    sns.barplot(x=list(values), y=list(features), color='skyblue')
    plt.title("特征重要性", fontproperties=my_font)
    plt.xlabel("Importance", fontproperties=my_font)
    plt.ylabel("Feature", fontproperties=my_font)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    if show:
        plt.show()
    else:
        plt.close()


