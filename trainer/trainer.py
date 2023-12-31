
from tqdm import tqdm
import os
import torch
from torch import nn

class Trainer():
    def __init__(self, model, training_loader, scheduler, logger, fastspeech_loss, optimizer, train_config):
        self.train_config = train_config
        self.model = model
        self.training_loader = training_loader
        self.logger = logger
        self.fastspeech_loss = fastspeech_loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        
    def train(self):
        current_step = 0
        tqdm_bar = tqdm(total=self.train_config.epochs * len(self.training_loader) * self.train_config.batch_expand_size - current_step)
        for epoch in range(self.train_config.epochs):
            for i, batchs in enumerate(self.training_loader):
                # real batch start here
                for j, db in enumerate(batchs):
                    current_step += 1
                    tqdm_bar.update(1)
                    
                    self.logger.set_step(current_step)

                    # Get Data
                    character = db["text"].long().to(self.train_config.device)
                    mel_target = db["mel_target"].float().to(self.train_config.device)
                    duration = db["duration"].int().to(self.train_config.device)
                    mel_pos = db["mel_pos"].long().to(self.train_config.device)
                    src_pos = db["src_pos"].long().to(self.train_config.device)
                    max_mel_len = db["mel_max_len"]

                    # Forward
                    mel_output, duration_predictor_output = self.model(character,
                                                                src_pos,
                                                                mel_pos=mel_pos,
                                                                mel_max_length=max_mel_len,
                                                                length_target=duration)
                    

                    # Calc Loss
                    mel_loss, duration_loss = self.fastspeech_loss(mel_output,
                                                            duration_predictor_output,
                                                            mel_target,
                                                            duration)
                    total_loss = mel_loss + duration_loss

                    # Logger
                    t_l = total_loss.detach().cpu().numpy()
                    m_l = mel_loss.detach().cpu().numpy()
                    d_l = duration_loss.detach().cpu().numpy()

                    self.logger.add_scalar("duration_loss", d_l)
                    self.logger.add_scalar("mel_loss", m_l)
                    self.logger.add_scalar("total_loss", t_l)

                    # Backward
                    total_loss.backward()

                    # Clipping gradients to avoid gradient explosion
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.train_config.grad_clip_thresh)
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scheduler.step()

                    if current_step % self.train_config.save_step == 0:
                        torch.save({'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(
                        )}, os.path.join(self.train_config.checkpoint_path, 'checkpoint_%d.pth.tar' % current_step))
                        print("save model at step %d ..." % current_step)
