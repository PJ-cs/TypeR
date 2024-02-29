import torch.nn as nn
import torch
from image2letterPackage.utils import load_letter_conv_weights
import tqdm


class CustomTransposedConv2d(nn.Module):
    def __init__(
        self,
        weights,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        out_padding=0,
    ):
        super(CustomTransposedConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.out_padding = out_padding
        self.weights = nn.Parameter(weights)

        # Define the hard-coded weights (example weights)
        # self.weights = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        # TODO normalize weights here
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        # Perform the transposed convolution operation with hard-coded weights
        output = nn.functional.conv_transpose2d(
            x,
            self.weights,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.out_padding,
        )
        return output


class LetterFilter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config["device"]
        self.letter_conv_k = config["letter_conv_k"]
        self.letter_conv_stride = config["letter_conv_stride"]
        font_path = config["font_path"]
        letters = config["letters"]
        self.num_letters = len(letters)
        self.letters_per_pix = config["letters_per_pix"]

        # eps is minimum value for letter match to be considered as such
        self.eps = torch.tensor(config["eps"]).to(self.device)

        self.zero = torch.tensor(0.0).to(self.device)
        self.one = torch.tensor(1.0).to(self.device)

        # overlap_gamma is the factor by which the letter match is subtracted from the input image
        self.overlap_gamma = torch.tensor(config["overlap_gamma"]).to(self.device)

        with torch.no_grad():
            # letter convolutions
            letter_conv_weights = load_letter_conv_weights(
                font_path, self.letter_conv_k, letters
            )
            self.letter_filter = nn.Conv2d(
                in_channels=1,
                out_channels=self.num_letters,
                kernel_size=self.letter_conv_k,
                stride=self.letter_conv_stride,
            )
            self.letter_filter.weight[:] = letter_conv_weights
            self.letter_filter.to(self.device)

            # transposed convs for mask
            transposed_convs_weights = load_letter_conv_weights(
                font_path, self.letter_conv_k, letters
            )
            transposed_padding = 0
            transpose_out_padding = self.letter_conv_stride - 1
            self.transp_conv = CustomTransposedConv2d(
                transposed_convs_weights,
                self.num_letters,
                1,
                self.letter_conv_k,
                self.letter_conv_stride,
                transposed_padding,
                transpose_out_padding,
            )
            self.transp_conv.to(self.device)

    def forward(self, input_img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            B = input_img.shape[0]

            # pad input image with half of letter conv kernel size
            input_img = nn.ReplicationPad2d(self.letter_conv_k // 2)(input_img)

            _, _, H_letter_hits, W_letter_hits = self.letter_filter(
                torch.randn_like(input_img).to(self.device)
            ).shape

            max_letter_hits_total = (
                B * H_letter_hits * W_letter_hits * self.letters_per_pix
            )

            letter_hits = torch.zeros(
                (B, self.num_letters, H_letter_hits, W_letter_hits)
            ).to(self.device)
            # letter hits total
            current_letter_hits = torch.zeros(1).to(self.device)

            # tracks letter hits pixels with max letter per pix reached, 0 = max reached
            filled_pixels_mask = torch.ones((B, 1, H_letter_hits, W_letter_hits)).to(
                self.device
            )

            letter_match = torch.zeros(
                (B, self.num_letters, H_letter_hits, W_letter_hits)
            ).to(self.device)
            current_img = input_img.clone()

            # letter matches
            letter_match: torch.Tensor = self.letter_filter(current_img)

            progress_bar = tqdm.tqdm(total=max_letter_hits_total, desc="Processing")

            while (
                torch.any(letter_match > self.eps)
                and current_letter_hits < max_letter_hits_total
            ):
                # fig, axes = plt.subplots(2, 5, figsize=(12,6))
                # axes.flat[0].imshow(grad_magnitude.detach().cpu()[0][0])
                # axes.flat[1].imshow(detail_map.detach().cpu()[0][0])
                # axes.flat[2].imshow(letter_match.detach().cpu()[0][0])
                # axes.flat[3].imshow(letter_match.detach().cpu()[0][33])
                # axes.flat[4].imshow(weighted_letter_match.detach().cpu()[0][33])
                # axes.flat[5].imshow(current_img.detach().cpu()[0][0])
                # axes.flat[5].imshow(self.transp_conv(letter_hits).clip(self.zero, self.one).detach().cpu()[0][0])

                # plt.tight_layout()
                # plt.show()

                index = torch.argmax(letter_match.view(B, -1), dim=1, keepdim=True)

                mask = torch.zeros_like(letter_match.view(B, -1))
                mask = mask.scatter(1, index, 1)
                mask = mask.view(B, self.num_letters, H_letter_hits, W_letter_hits)
                mask[letter_match < self.eps] = self.zero
                current_letter_hits += torch.sum(mask > self.zero)

                # add max letter hit of image of current iteration to all letterhits
                letter_hits = letter_hits + mask.mul(letter_match)

                # set filled pixels to 0 in filled pixels mask
                filled_pixels_mask = (letter_hits > self.zero).sum(
                    dim=1, keepdim=True
                ) < self.letters_per_pix

                # update current image: substract found best letter from input img TODO make this negative?
                current_img = input_img - self.overlap_gamma * self.transp_conv(
                    letter_hits
                )

                # letter matches
                letter_match = self.letter_filter(current_img)
                # set pixel with reached max letters per pix to zero
                letter_match = letter_match.mul(filled_pixels_mask)

                progress_bar.update(1)

            return letter_hits, self.transp_conv(letter_hits)
