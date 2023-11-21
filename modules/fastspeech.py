from torch import nn
import torch
from modules.encoder import Encoder
from modules.lengthregulator import LengthRegulator
from modules.decoder import Decoder

class FastSpeech(nn.Module):
    """ Fast Speech """

    def __init__(self, model_config, mel_config):
        super(FastSpeech, self).__init__()

        self.encoder = Encoder(model_config)
        self.length_regulator = LengthRegulator(model_config)
        self.decoder = Decoder(model_config)

        self.mel_linear = nn.Linear(model_config.decoder_dim, mel_config.num_mels)

    def get_mask_from_lengths(self, lengths, max_len=None):
        if max_len == None:
            max_len = torch.max(lengths).item()

        ids = torch.arange(0, max_len, 1, device=lengths.device)
        mask = (ids < lengths.unsqueeze(1)).bool()

        return mask

    def mask_tensor(self, mel_output, position, mel_max_length):
        lengths = torch.max(position, -1)[0]
        mask = ~self.get_mask_from_lengths(lengths, max_len=mel_max_length)
        mask = mask.unsqueeze(-1).expand(-1, -1, mel_output.size(-1))
        return mel_output.masked_fill(mask, 0.)

    def forward(self, src_seq, src_pos, mel_pos=None, mel_max_length=None, length_target=None, alpha=1.0):
        encoder_output, _ = self.encoder(src_seq, src_pos)

        if self.training:
            length_regulator_output, duration_predictor_output = self.length_regulator(encoder_output,
                                                                                       target=length_target,
                                                                                       alpha=alpha,
                                                                                       mel_max_length=mel_max_length)
            decoder_output = self.decoder(length_regulator_output, mel_pos)

            mel_output = self.mel_linear(decoder_output)
            mel_output = self.mask_tensor(mel_output, mel_pos, mel_max_length)

            return mel_output, duration_predictor_output
        else:
            length_regulator_output, decoder_pos = self.length_regulator(encoder_output,
                                                                         alpha=alpha)

            decoder_output = self.decoder(length_regulator_output, decoder_pos)

            mel_output = self.mel_linear(decoder_output)
            return mel_output
