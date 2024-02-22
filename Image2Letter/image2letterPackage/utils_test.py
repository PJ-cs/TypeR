import os
from image2letterPackage.utils import load_letter_conv_weights
import json
import torch
import matplotlib.pyplot as plt


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
        #  assert that the sum of the convolutions is 1 per convolution
        assert torch.allclose(actual.sum(dim=(1, 2, 3)), torch.ones(len(letters)))

        # plot the actual convolutions with their respective letters + value scale and save the plot
        fig, axs = plt.subplots(5, len(letters) // 5, figsize=(60, 10))
        for i, ax in enumerate(axs):
            for j, ax2 in enumerate(ax):
                ax2.imshow(actual[i * 5 + j, 0], cmap="gray")
                ax2.set_title(
                    letters[i * 5 + j]
                    + " min: "
                    + str(round(actual[i * 5 + j].min().item(), 3))
                    + " max: "
                    + str(round(actual[i * 5 + j].max().item(), 3))
                )

        plt.tight_layout()
        test_name: str = os.getenv("PYTEST_CURRENT_TEST").split(" ")[-2].replace("/", "_")
        plt.savefig(
            os.path.join(os.getenv("UNIT_TESTING_IMGS_DIR"), test_name + ".png")
        )
