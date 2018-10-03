import pylab

unidirectional_history_file = '../Neural_unidirectional_history_29-08-2018_22-34-53'
bidirectional_history_file = '../Neural_bidirectional_history_30-08-2018_00-46-38'
elmo_history_file = '../Elmo_history_02-10-2018_07-13-24'

epochs_text = 'Przebieg sieci'
loss_text = 'Wartość entropii krzyżowej'
pylab.xlabel(epochs_text)
pylab.ylabel(loss_text)

with open(unidirectional_history_file) as data_fl:
    un_losses = [float(point) for point in data_fl.read().strip().split(' ')]
    un_epochs = range(1, len(un_losses)+1)
    pylab.plot(un_epochs, un_losses, label='LSTM-1')

with open(bidirectional_history_file) as data_fl:
    un_losses = [float(point) for point in data_fl.read().strip().split(' ')]
    un_epochs = range(1, len(un_losses)+1)
    pylab.plot(un_epochs, un_losses, label='LSTM-2')

with open(elmo_history_file) as data_fl:
    un_losses = [float(point) for point in data_fl.read().strip().split(' ')]
    un_epochs = range(1, len(un_losses)+1)
    pylab.plot(un_epochs, un_losses, label='LSTM-ELMo')

pylab.legend(loc='upper right')
pylab.show()
