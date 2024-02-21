import dotenv
import os
from image2letterPackage.utils import load_letter_conv_weights
import json
import torch

dotenv.load_dotenv(dotenv_path=".env.dev")


class TestLoad_letter_conv_weights:
    def setup_method(self):
        self.font_path = os.getenv("FONT_PATH")
        with open(os.getenv("TYPEWRITER_CONFIG_PATH")) as f:
            self.typewriter_config = json.load(f)

    def test_count_convs(self):
        letters = self.typewriter_config["SamsungSQ-1000"]["letterList"]
        kernel_size = 28
        actual = load_letter_conv_weights(self.font_path, kernel_size, letters=letters)
        assert actual.shape == (len(letters), 1, 28, 28)
        assert actual.dtype == torch.float32
        assert actual.device.type == "cpu"
        assert actual.requires_grad is False
        assert actual.min() >= 0.0 and actual.max() <= 1.0
