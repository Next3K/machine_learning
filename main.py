import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from sklearn.metrics import ConfusionMatrixDisplay
matplotlib.use('TkAgg')


def get_file(name):
    file = open(name + ".txt", 'r')
    content = file.read()
    content = content.replace("[", "").replace("]", "")
    lines = content.strip().split("\n")
    matrix_data = [list(map(int, line.split())) for line in lines[:-2]]
    float1 = float(lines[-2])
    float2 = float(lines[-1])

    matrix = np.array(matrix_data)

    return matrix, float1, float2


def check_dir():
    if not os.path.exists("charts"):
        os.makedirs("charts")


def confusion_matrix_display(matrix, title):
    check_dir()
    disp = ConfusionMatrixDisplay(matrix, display_labels=["0","1", "2", "3", "4", "5"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"{title}")
    plt.savefig(f"charts/{title}.png", dpi=300)
    plt.cla()
    plt.close()


if __name__ == '__main__':
    matrix = get_file("feedback_1")[0]
    confusion_matrix_display(matrix, "kNN")
    matrix = get_file("feedback_2")[0]
    confusion_matrix_display(matrix, "Decision trees")
    matrix = get_file("feedback_3")[0]
    confusion_matrix_display(matrix, "Random forrest")
    matrix = get_file("feedback_4")[0]
    confusion_matrix_display(matrix, "Person similarity")
    matrix = get_file("feedback_6")[0]
    confusion_matrix_display(matrix, "Person similarity - dodatkowy test I")
    matrix = get_file("feedback_7")[0]
    confusion_matrix_display(matrix, "Person similarity - dodatkowy test II")
    matrix = get_file("feedback_forest_test")[0]
    confusion_matrix_display(matrix, "Forest (train)")
    matrix = get_file("feedback_forest_validation")[0]
    confusion_matrix_display(matrix, "Forest (validation)")
    matrix = get_file("feedback_trees_test")[0]
    confusion_matrix_display(matrix, "Trees (train)")
    matrix = get_file("feedback_trees_validation")[0]
    confusion_matrix_display(matrix, "Trees (validation)")
    matrix = get_file("feedback_knn_test")[0]
    confusion_matrix_display(matrix, "KNN (train)")
    matrix = get_file("feedback_knn_validation")[0]
    confusion_matrix_display(matrix, "KNN (validation)")
