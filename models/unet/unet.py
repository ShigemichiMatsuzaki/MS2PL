import torch
import torch.nn as nn
import torch.nn.functional as F


class CRx2(nn.Module):
    def __init__(self, in_channel, out_channel, mid_channel=None):
        """Conv + ReLU"""
        super(CRx2, self).__init__()
        if mid_channel is None:
            mid_channel = in_channel

        self.sequential = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """Forward pass"""
        out = self.sequential(x)

        return out


class UNet(nn.Module):
    def __init__(self, num_classes):
        """UNet"""
        super(UNet, self).__init__()

        self.crx2_enc_1 = CRx2(3, 64, 64)
        self.crx2_enc_2 = CRx2(64, 128)
        self.crx2_enc_3 = CRx2(128, 256)
        self.crx2_enc_4 = CRx2(256, 512)
        self.crx2_enc_5 = CRx2(512, 512)

        self.crx2_dec_4 = CRx2(1024, 256)
        self.crx2_dec_3 = CRx2(512, 128)
        self.crx2_dec_2 = CRx2(256, 64)
        self.crx2_dec_1 = CRx2(128, 64)

        self.classifier_main = nn.Conv2d(64, num_classes, kernel_size=1, bias=False)
        self.classifier_aux = nn.Conv2d(64, num_classes, kernel_size=1, bias=False)

        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x):
        x_size = (x.size(2), x.size(3))

        ### DOWN ###
        out_enc_1 = self.crx2_enc_1(x)
        out_enc_1_ds = self.max_pool(out_enc_1)

        out_enc_2 = self.crx2_enc_2(out_enc_1_ds)
        out_enc_2_ds = self.max_pool(out_enc_2)

        out_enc_3 = self.crx2_enc_3(out_enc_2_ds)
        out_enc_3_ds = self.max_pool(out_enc_3)

        out_enc_4 = self.crx2_enc_4(out_enc_3_ds)
        out_enc_4_ds = self.max_pool(out_enc_4)

        out_enc_5 = self.crx2_enc_5(out_enc_4_ds)
        # out_dec_5_us = self.upsample(out_enc_5)

        ### UP ###
        in_dec_4 = torch.cat([out_enc_4_ds, out_enc_5], dim=1)
        out_dec_4 = self.crx2_dec_4(in_dec_4)
        out_dec_4_us = self.upsample(out_dec_4)

        in_dec_3 = torch.cat([out_enc_3_ds, out_dec_4_us], dim=1)
        out_dec_3 = self.crx2_dec_3(in_dec_3)
        out_dec_3_us = self.upsample(out_dec_3)

        in_dec_2 = torch.cat([out_enc_2_ds, out_dec_3_us], dim=1)
        out_dec_2 = self.crx2_dec_2(in_dec_2)
        out_dec_2_us = self.upsample(out_dec_2)

        in_dec_1 = torch.cat([out_enc_1_ds, out_dec_2_us], dim=1)
        out_dec_1 = self.crx2_dec_1(in_dec_1)

        main_out = self.classifier_main(out_dec_1)
        aux_out = self.classifier_main(out_dec_2)

        return {
            "out": F.interpolate(
                main_out, size=x_size, mode="bilinear", align_corners=True
            ),
            "aux": F.interpolate(
                aux_out, size=x_size, mode="bilinear", align_corners=True
            ),
        }
