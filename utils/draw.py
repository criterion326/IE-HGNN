import datetime
def draw_losses(losses, title=['', ''], save_dir='./'):
    plt.figure(figsize=(6, 7))
    plt.xlabel('epoches', fontsize=10)
    plt.ylabel('Loss', fontsize=10)
    for i in range(len(title)):
        plt.subplot(2, 1, i + 1)
        plt.plot(list(range(1, len(losses[i]) + 1)), losses[i])
        plt.title(title[i])
    filetime = str(
        datetime.datetime.strftime(datetime.datetime.today(), '%Y%m%d_%H%M%S'))
    plt.savefig(os.path.join(save_dir, 'loss_' + filetime + '.jpg'))
    plt.show()
