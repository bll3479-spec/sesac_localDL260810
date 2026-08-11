from torchvision.models import vgg11, VGG11_Weights

#모델 로딩
def get_vgg_model():
    model = vgg11(weights = VGG11_Weights.DEFAULT)
    #모델의 기억을 일부 동결, 일부 학습용으로 비워놓음
    #feautures 파트 가중치는 '학습불가'로 동결, classifier만 수정.
    #전체 가중치 동결
    for params in model.parameters():       
        params.requires_grad=False
    #분류기만 동결 해제
    for params in model.classifier.parameters():
        params.requires_grad = True
    return model