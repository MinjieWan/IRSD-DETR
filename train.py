import warnings, os
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR(R'C:\Users\LENOVO\Desktop\IRSD-DETR_demo\IRSD-DETR.yaml')
    model.train(data=R'C:\Users\LENOVO\Desktop\IRSD-DETR_demo\SDSS\SDSS.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=8,
                workers=2,
                patience=0,
                project='runs/train',
                name='exp',
                )