from torch import nn
from modules.durationpredictor import DurationPredictor
import torch
from utils import create_alignment
import torch.nn.functional as F

class LengthRegulator(nn.Module):
    """ Length Regulator """

    def __init__(self, model_config):
        super(LengthRegulator, self).__init__()
        self.duration_predictor = DurationPredictor(model_config)

    def LR(self, x, duration_predictor_output, mel_max_length=None):
        expand_max_len = torch.max(
            torch.sum(duration_predictor_output, -1), -1)[0]
        alignment = torch.zeros(duration_predictor_output.size(0),
                                expand_max_len,
                                duration_predictor_output.size(1)).numpy()
        alignment = create_alignment(alignment,
                                     duration_predictor_output.cpu().numpy())
        alignment = torch.from_numpy(alignment).to(x.device)

        output = alignment @ x
        if mel_max_length:
            output = F.pad(
                output, (0, 0, 0, mel_max_length-output.size(1), 0, 0))
        return output

    def forward(self, x, alpha=1.0, target=None, mel_max_length=None):
        duration_predictor_output = self.duration_predictor(x)

        if target is not None:
            output = self.LR(x, target, mel_max_length=mel_max_length)
            return output, duration_predictor_output
        else:
            duration_predictor_output = (
                (duration_predictor_output + 0.5) * alpha).int()
            output = self.LR(x, duration_predictor_output)
            mel_pos = torch.stack(
                [torch.Tensor([i for i in range(1, output.size(1) + 1)])]).long().to(x.device)

            return output, mel_pos