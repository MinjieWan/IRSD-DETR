import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
from prettytable import PrettyTable
from ultralytics import RTDETR
from ultralytics.utils.torch_utils import model_info

def get_weight_size(path):
    stats = os.stat(path)
    return f'{stats.st_size / 1024 / 1024:.1f}'

if __name__ == '__main__':
    model_path = r'C:\Users\LENOVO\Desktop\IRSD-DETR_demo\IRSD-DETR.pt'     #load the weight
    model = RTDETR(model_path)
    result = model.val(data=r'C:\Users\LENOVO\Desktop\IRSD-DETR_demo\SDSS\SDSS.yaml',   # load the dataset config
                      split='val',
                      imgsz=640,
                      batch=1,
                      project='runs/val',
                      name='exp',
                      )

    if model.task == 'detect':
        length = result.box.p.size
        model_names = list(result.names.values())

        model_metrice_table = PrettyTable()
        model_metrice_table.title = "Model Metrice"

        model_metrice_table.field_names = ["Class Name", "Precision", "Recall", "F1-Score", "mAP50", "mAP50-95"]

        for idx in range(length):
            model_metrice_table.add_row([
                model_names[idx],
                f"{result.box.p[idx]:.4f}",
                f"{result.box.r[idx]:.4f}",
                f"{result.box.f1[idx]:.4f}",
                f"{result.box.ap50[idx]:.4f}",
                # Removed the mAP75 data line here
                f"{result.box.ap[idx]:.4f}"
            ])

        model_metrice_table.add_row([
            "all(平均数据)",
            f"{result.results_dict['metrics/precision(B)']:.4f}",
            f"{result.results_dict['metrics/recall(B)']:.4f}",
            f"{np.mean(result.box.f1[:length]):.4f}",
            f"{result.results_dict['metrics/mAP50(B)']:.4f}",
            f"{result.results_dict['metrics/mAP50-95(B)']:.4f}"
        ])
        print(model_metrice_table)


