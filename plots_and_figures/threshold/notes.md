### Thresholding


The threshold happends with a sigmoid. 

$w_1 = 1$ and $w_2(c) = 1 - sigmoid((c - \alpha)/\tau)$.

- $c$ is the value of the criterion image computed from norm of the spatial gradient of the PAN image normalized as to sum to one.

- $\alpha$ is a threshold automatically tuned by applying Otsu's algorithm. 
- $\tau$ is a smoothness parameter on the sigmoid. The smallest the $\tau$ the more the sigmoid behaves as a hard-threshold. When $\tau$ is too large compared to the range of the criterion, the sigmoid is close to 1/2.

### Noise on images ?

- The tuning depends on the noise level. We experiment with 3 noise level :  
    - low : 50dB 
    - mid : 40dB
    - high : 37dB

- The impact of the noise can be seen on the criterion histogram.

### Tuning the smoothness parameter $\tau$

- The behavior of the thresholding across $\tau$ seems not to depend on the image (tested on 4 images)
- We choose 3 values of $\tau$ that exhibit more or less smoothness : 
    - 4e-5 (good with low noise)
    - 2e-5 (makes the edges appear even when a little bit of noise ?)
    - 1e-5 (behaves like a hard threshold, more robust when noisy ?)