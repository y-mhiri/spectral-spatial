import torch
from deepinv.physics import Blur, Downsampling, GaussianNoise, Denoising
import zarr
from torch.utils import data
import deepinv as dinv


class PANDataset(data.Dataset):
    """
    Hyperspectral pansharpening dataset. Loads images from a Zarr archive and
    simulates the degradation operators (blur + downsample for HSI, spectral
    averaging for PAN) used in the pansharpening forward model.

    Attributes:
        nband (int):   Number of spectral bands.
        height (int):  Image height.
        width (int):   Image width.
        scale (int):   Spatial downsampling factor.
        noise_level:   Noise standard deviation (converted from dB: 10^(-dB/20)).
        blur_op:       Gaussian blur operator (deepinv).
        downsample_op: Downsampling operator (deepinv).
    """

    def __init__(self, root_dir, split='train', transform=None, normalize=False,
                 scale=4, sigma_blur=0.001, noise_level=0.001, device="cpu", seed=0):
        """
        Args:
            root_dir (str):        Path to the Zarr file.
            split (str):           Data partition ('train'/'test'/'val').
            transform (callable):  Optional transform applied to each image.
            normalize (bool):      Normalize to [0,1] if True.
            scale (int):           Spatial downsampling factor.
            sigma_blur (float):    Gaussian blur sigma.
            noise_level (float):   Noise level in dB; converted as 10^(-noise_level/20).
            device (str):          Compute device.
        """
        super().__init__()
        self.file      = zarr.open(root_dir, mode='r')
        self.split     = split
        self.transform = transform
        self.normalize = normalize
        self.scale     = scale
        self.sigma_blur = sigma_blur
        self.device    = device
        self.seed      = seed

        # Metadata
        self.rgb_index          = self.file.attrs['rgb']
        self.wavenumbers        = self.file.attrs['spectral_range']
        self.spatial_resolution  = self.file.attrs['spatial_resolution (m)']
        self.spectral_resolution = self.file.attrs['spectral_resolution (nm)']

        # Dimensions — use .shape to avoid loading data from disk
        shape = self.file[self.split][str(0)].shape   # (H, W, C)
        self.height = shape[0]
        self.width  = shape[1]
        self.nband  = shape[2]
        img_size = (self.nband, self.height, self.width)

        self.noise_level = 10 ** (-noise_level / 20)

        self.R = (1 / self.nband) * torch.ones(1, self.nband, device=self.device)

        self.blur_op = Blur(
            filter=dinv.physics.blur.gaussian_blur(sigma=(self.sigma_blur, self.sigma_blur), angle=0.0),
            padding='circular',
            device=self.device
        )
        self.downsample_op = Downsampling(
            img_size=img_size,
            filter='gaussian',
            factor=self.scale,
            padding='circular',
            device=self.device
        )

        # Noise model cached at construction time
        noise_model = GaussianNoise(self.noise_level)
        self._noise_physics = Denoising(noise_model=noise_model)

    def __len__(self):
        return len(self.file[self.split])

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Sample index.

        Returns:
            torch.Tensor: Hyperspectral image [C, H, W].
        """
        img = torch.from_numpy(self.file[self.split][str(idx)][:]).float()
        img = img.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)

        if self.transform:
            img = self.transform(img)

        if self.normalize:
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        return img.to(self.device)

    def get_operators(self):
        A     = lambda x: self.downsample_op(self.blur_op(x))
        A_adj = lambda x: self.blur_op(self.downsample_op.A_adjoint(x))
        R     = self.spectral_op
        R_adj = self.spectral_op_t
        return A, A_adj, R, R_adj

    def simulate_low_res_hsi(self, input_image, noise=True):
        """
        Simulate a low-resolution HSI acquisition (blur + downsample).

        Args:
            input_image (torch.Tensor): HR image [b, c, h, w].

        Returns:
            torch.Tensor: LR image [b, c, h//scale, w//scale].
        """
        if input_image.ndim != 4:
            raise ValueError("input_image must be a 4D tensor [b, c, h, w]")
        lr = self.downsample_op(self.blur_op(input_image))
        return self._noise_physics(lr) if noise else lr

    def simulate_panchromatic(self, input_image, noise=True):
        """
        Compute the panchromatic image by spectral averaging.

        Args:
            input_image (torch.Tensor): Hyperspectral image [b, c, h, w].

        Returns:
            torch.Tensor: Panchromatic image [b, 1, h, w].
        """
        pan = self.spectral_op(input_image)
        return self._noise_physics(pan) if noise else pan

    def spectral_op(self, input_image):
        """
        Spectral averaging operator R: uniform mean across bands.

        Args:
            input_image (torch.Tensor): Hyperspectral image [b, c, h, w].

        Returns:
            torch.Tensor: Panchromatic image [b, 1, h, w].
        """
        b, c, h, w = input_image.shape
        X  = input_image.reshape(b, c, -1)       # [b, c, h*w]
        RX = torch.matmul(self.R, X)             # [b, 1, h*w]
        return RX.reshape(b, 1, h, w)

    def spectral_op_t(self, input_image):
        """
        Adjoint spectral operator R^T: broadcast single-band image to all bands.

        Args:
            input_image (torch.Tensor): Single-band image [b, 1, h, w].

        Returns:
            torch.Tensor: Hyperspectral image [b, c, h, w].
        """
        b, _, h, w = input_image.shape
        Y   = input_image.reshape(b, 1, -1)      # [b, 1, h*w]
        RtX = torch.matmul(self.R.t(), Y)        # [b, c, h*w]
        return RtX.reshape(b, self.nband, h, w)

    def noise(self, input_image):
        return self._noise_physics(input_image)
