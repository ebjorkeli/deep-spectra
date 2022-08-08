from DataGenerator.FFT_data_generator import*
from Models.FFT_model_2D import model_1
from Trainers.FFT_trainer import Trainer




def main():
    # Data loader
    train_data = DataGenerator(N=100)
    val_data   = ValGenerator(N=5)

    # Set up model
    dim = train_data.dim   #train_data[0].shape # N*t*2 (N,1024,2)
    in_shape = dim[1:]
    out_shape = (dim[1],1)

    model = model_1(in_shape, out_shape)

    # Hyperparameters to run over
    eps = [100] #[200]
    bss = [10]#, 20, 50, 100, 200]
    lrs = [5e-6]# , 1e-6, 5e-5, 1e-5, 5e-4, 1e-4, 5e-3, 1e-3]

    # Looping trainer
    for ep in eps:
        for bs in bss:
            for lr in lrs:
                trainer = Trainer(model, train_data, val_data, ep, bs, lr, save_model=False)
                loss = trainer.train()

if __name__ == '__main__':
    main()