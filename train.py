from dataset.buffer_dataset import BufferDataset
from utils import get_data_to_buffer
from torch.utils.data import DataLoader
from collate.collate import collate_fn_tensor
from modules.fastspeech import FastSpeech
from loss.fastspeechloss import FastSpeechLoss
import torch
from wandb_writer import WanDBWriter
from trainer.trainer import Trainer
from torch.optim.lr_scheduler import OneCycleLR
from configs.config import MelSpectrogramConfig, FastSpeechConfig, TrainConfig


def main():
    mel_config = MelSpectrogramConfig()
    model_config = FastSpeechConfig()
    train_config = TrainConfig()

    buffer = get_data_to_buffer(train_config)

    dataset = BufferDataset(buffer)

    training_loader = DataLoader(
        dataset,
        batch_size=train_config.batch_expand_size * train_config.batch_size,
        shuffle=True,
        collate_fn=collate_fn_tensor,
        drop_last=True,
        num_workers=0
    )

    model = FastSpeech(model_config)
    model = model.to(train_config.device)

    fastspeech_loss = FastSpeechLoss()
    

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        betas=(0.9, 0.98),
        eps=1e-9)

    scheduler = OneCycleLR(optimizer, **{
        "steps_per_epoch": len(training_loader) * train_config.batch_expand_size,
        "epochs": train_config.epochs,
        "anneal_strategy": "cos",
        "max_lr": train_config.learning_rate,
        "pct_start": 0.1
    })

    logger = WanDBWriter(train_config)

    trainer = Trainer(
        moodel=model,
        training_loader=training_loader,
        scheduler=scheduler,
        logger=logger,
        fastspeech_loss=fastspeech_loss,
    )

    trainer.train()

if __name__ == '__main__':
    main()