import os
import matplotlib.pyplot as plt


#매개변수에 '=': 매개변수=값. 매개변수를 따로 지정하지 않아도 default 값 지정. 꼭 항상 마지막에 위치.
def draw_plot(history, save_path = r'./result.jpg'):
    #히스토리 길이 체크
    #history(train_loss, train_acc, valid_loss, valid_acc)
    epochs = range(1, len(history['train_acc'])+1)

    fig, ax = plt.subplots(1, 2, figsize=(14,6))

    ax[0].plot(epochs, history['train_loss'], label = 'TR_loss')
    ax[0].plot(epochs, history['valid_loss'], label = 'VL_loss')
    ax[0].legend()      #ax[0].set_title('Loss')

    ax[1].plot(epochs, history['train_acc'], label = 'TR_acc')
    ax[1].plot(epochs, history['valid_acc'], label = 'VL_acc')
    ax[1].legend()      #ax[1].set_title('ACC(%)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f'저장 완료: {save_path}에 저장됨')
