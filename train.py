


from dataset.cocoDatasets import create_dataloader
from models.YOLOv3 import load_model
import yaml
import torch 

if __name__ == "__main__":
    train_path = "/home/wu/Desktop/yolov3-copy/data/coco/trainvalno5k.txt"
    with open('hyp.scratch.yaml') as f:    
        hyp = yaml.load(f, Loader=yaml.FullLoader)  # load hyps
    dataloader, dataset = create_dataloader(train_path, imgsz=416, batch_size=4, stride=8, opt=None, hyp=hyp, augment=True)


    #
    model = load_model("./config/yolov3.cfg")
    #print(model)
    device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = [p for p in model.parameters() if p.requires_grad]





    for i, (imgs, targets, paths, _) in enumerate(dataloader): 
        imgs = imgs.to("cpu").float() / 255.0
        #print(imgs.dtype)

        output = model(imgs)
        #print(imgs.numpy()[0].reshape(640, 640, 3).shape)