from torchvision.models import resnet34, ResNet34_Weights

def get_resnet_model():
    model = resnet34(weights = ResNet34_Weights.DEFAULT)
    #전체 가중치 동결
    for p in model.parameters():
        p.requires_grad=False
    #fc 동결 해제
    for params in model.fc.parameters():
        params.requires_grad=True    

    return model