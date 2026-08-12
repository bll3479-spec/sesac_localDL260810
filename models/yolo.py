from ultralytics import YOLO

def yolo_train():
    
    # train = r'C:\Users\user\Desktop\Git\sesac_localDL260810\Data\peach_dataset\YoloDataset\labels\valid'
    # paths = [os.path.join(train, x) for x in os.listdir(train)]

    # for i in paths:
    #     t = i.split('\\')[-1] + '.txt'
    #     os.rename(i, os.path.join(train, t))


    #YOLO 라이브러리 세팅(pip)
    yaml_path = r'./yolo_setting.yaml'
    #YOLO 훈련
    YOLO('yolov8n.pt').train(
                data=yaml_path,  
                epochs = 10, 
                imgsz = 640, 
                batch=16, 
                save = True,
                device = 0, 
                plots = True,
                name = 'peach_train02')
    print('훈련 완료')

  