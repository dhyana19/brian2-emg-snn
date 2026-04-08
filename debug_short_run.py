import sys, os
print('Starting short debug training run (N_TRAIN=20, EPOCHS=1)')
# Ensure src is importable
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
import train_snn
train_snn.N_TRAIN = 20
train_snn.EPOCHS = 1
train_snn.train_snn()
print('Debug run complete')
