import os
import json
import torch
from image2letterPackage.models import CustomTransposedConv2d, LetterFilter
from image2letterPackage.utils import load_letter_conv_weights
import matplotlib.pyplot as plt


class TestModels:
    def setup_method(self):
        self.font_path = os.getenv("FONT_PATH")
        with open(os.getenv("TYPEWRITER_CONFIG_PATH")) as f:
            self.typewriter_config = json.load(f)
        self.letters = self.typewriter_config["SamsungSQ-1000"]["letterList"]

    def test_model_forward(self):
        config = {
            "device": "cpu",
            "letter_conv_k": 31,
            "letter_conv_stride": 1,
            "font_path": self.font_path,
            "letters": self.letters,
            "letters_per_pix": 2,
            "eps": 0.1,
            "letter_size_weight": 0.0,
            "detail_weight": 0.0,
            "overlap_gamma": 1.0,
        }
        letter_filter_sut = LetterFilter(config)
        conv_weights = load_letter_conv_weights(
            self.font_path, config["letter_conv_k"], letters=self.letters
        )

        input_img = torch.zeros(1, 1, 100, 100)
        letter_Z_weight = conv_weights[34]
        input_img[0, 0, 10:10 + 31, 10:10 + 31] = letter_Z_weight[::]
        expected_output_img = input_img.clone()
        expected_letter_hits = torch.zeros(1, len(self.letters), 100, 100)
        actual_letter_hits, actual_output_img= letter_filter_sut.forward(input_img)
        assert actual_output_img == expected_output_img
        assert actual_letter_hits[0, 34]

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
        test_name: str = (
            os.getenv("PYTEST_CURRENT_TEST").split(" ")[-2].replace("/", "_")
        )
        plt.savefig(
            os.path.join(os.getenv("UNIT_TESTING_IMGS_DIR"), test_name + ".png")
        )
