import torch
import torch.nn.functional as F


# ======================================================
# 3D Sobel Kernel (Correct 3¡Á3¡Á3)
# ======================================================

def get_sobel_3d(device, dtype):

    sobel_x = torch.tensor(
        [[[[-1, 0, 1],
           [-2, 0, 2],
           [-1, 0, 1]],

          [[-2, 0, 2],
           [-4, 0, 4],
           [-2, 0, 2]],

          [[-1, 0, 1],
           [-2, 0, 2],
           [-1, 0, 1]]]],
        dtype=dtype, device=device
    )

    sobel_y = torch.tensor(
        [[[[-1, -2, -1],
           [ 0,  0,  0],
           [ 1,  2,  1]],

          [[-2, -4, -2],
           [ 0,  0,  0],
           [ 2,  4,  2]],

          [[-1, -2, -1],
           [ 0,  0,  0],
           [ 1,  2,  1]]]],
        dtype=dtype, device=device
    )

    sobel_z = torch.tensor(
        [[[[-1, -2, -1],
           [-2, -4, -2],
           [-1, -2, -1]],

          [[ 0,  0,  0],
           [ 0,  0,  0],
           [ 0,  0,  0]],

          [[ 1,  2,  1],
           [ 2,  4,  2],
           [ 1,  2,  1]]]],
        dtype=dtype, device=device
    )

    return sobel_x.unsqueeze(0), sobel_y.unsqueeze(0), sobel_z.unsqueeze(0)


# ======================================================
# Utility: Ensure 5D tensor
# ======================================================

def ensure_5d(x):

    x = torch.as_tensor(x)

    if x.dim() == 4:
        x = x.unsqueeze(1)

    elif x.dim() == 5:
        pass

    elif x.dim() == 6:
        # remove accidental extra channel dim
        x = x.squeeze(1)

    else:
        raise ValueError(f"Unexpected tensor shape: {x.shape}")

    return x.contiguous()


# ======================================================
# Differentiable Boundary from Segmentation
# ======================================================

def extract_prob_boundary(logits, class_a=0, class_b=1):

    logits = torch.as_tensor(logits)

    prob = torch.softmax(logits, dim=1)
    diff = prob[:, class_a:class_a+1] - prob[:, class_b:class_b+1]

    diff = ensure_5d(diff)

    device = diff.device
    dtype = diff.dtype

    sobel_x, sobel_y, sobel_z = get_sobel_3d(device, dtype)

    dx = F.conv3d(diff, sobel_x, padding=1)
    dy = F.conv3d(diff, sobel_y, padding=1)
    dz = F.conv3d(diff, sobel_z, padding=1)

    grad = torch.sqrt(dx**2 + dy**2 + dz**2 + 1e-6)

    max_val = torch.amax(grad, dim=(2,3,4), keepdim=True) + 1e-6
    grad = grad / max_val

    return grad


# ======================================================
# Hard Boundary from Supervoxel
# ======================================================

def extract_supervoxel_boundary(supervoxel):

    sv = ensure_5d(supervoxel).float()

    device = sv.device
    dtype = sv.dtype

    sobel_x, sobel_y, sobel_z = get_sobel_3d(device, dtype)

    dx = F.conv3d(sv, sobel_x, padding=1)
    dy = F.conv3d(sv, sobel_y, padding=1)
    dz = F.conv3d(sv, sobel_z, padding=1)

    grad = torch.sqrt(dx**2 + dy**2 + dz**2)

    boundary = (grad > 0).float()

    return boundary


# ======================================================
# Gaussian Blur
# ======================================================

def gaussian_kernel_3d(kernel_size=5, sigma=1.5, device="cuda", dtype=torch.float32):

    ax = torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2
    zz, yy, xx = torch.meshgrid(ax, ax, ax, indexing="ij")

    kernel = torch.exp(-(xx**2 + yy**2 + zz**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()

    return kernel.unsqueeze(0).unsqueeze(0)


def gaussian_blur_3d(x, kernel_size=5, sigma=1.5):

    x = ensure_5d(x)

    device = x.device
    kernel = gaussian_kernel_3d(kernel_size, sigma, device, x.dtype)

    padding = kernel_size // 2

    return F.conv3d(x, kernel, padding=padding)


# ======================================================
# Final Boundary Loss
# ======================================================

def boundary_loss(
    logits,
    supervoxel,
    class_a=0,
    class_b=1,
    kernel_size=5,
    sigma=1.5,
    loss_type="smooth_l1"
):

    pred_boundary = extract_prob_boundary(logits, class_a, class_b)

    with torch.no_grad():
        sv_boundary = extract_supervoxel_boundary(supervoxel)
        sv_boundary = gaussian_blur_3d(sv_boundary, kernel_size, sigma)

    if loss_type == "mse":
        loss = F.mse_loss(pred_boundary, sv_boundary)
    else:
        loss = F.smooth_l1_loss(pred_boundary, sv_boundary)

    return loss
