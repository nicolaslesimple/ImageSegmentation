
import conv_net
from flip_training import flip_training


def run():
    flip_training()
    conv_net.execute(restore_flag = False)






if __name__ == '__main__':
    run()
