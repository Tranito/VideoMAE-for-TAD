import matplotlib.pyplot as plt
import wandb
import torch
from PIL import Image
import io
import torchvision.transforms as transforms

def create_bar_chart(accuracy_per_label, log_prefix, logger, bin_width=1):
    """
    Create and log a bar chart showing accuracy per label/bin.

    Args:
        accuracy_per_label: Tensor of accuracies for each label
        log_prefix: Prefix for logging (e.g., "val", "test")
        logger: PyTorch Lightning logger (e.g., self.trainer.logger)
        bin_width: Width of each bin in seconds
    """
    fig, ax = plt.subplots(figsize=(len(accuracy_per_label), 4))

    # Create color array - orange for all bars except last one (grey + transparent)
    colors = ['grey'] + ['orange'] * (len(accuracy_per_label) - 1) 
    labels = [str(label) for label in range(0, len(accuracy_per_label))]
    bars = plt.bar(labels[::-1], torch.flip(accuracy_per_label, dims=(0,)).numpy(), width=0.3, color=colors)
    plt.ylabel("Accuracy [%]", fontsize=14)
    plt.xlabel("Bin (Collision within [X]s)", fontsize=14)
    plt.title("Bin accuracy")
    plt.xticks(fontsize=10, rotation=90)
    plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], fontsize=12)
    if bin_width % 1 == 0:
        plt.xticks(labels, [f"{(bin_width + i*bin_width)}s" for i in range(len(labels)-1)] + ["No Coll. Soon"], fontsize=12, rotation=30)
    else:
        plt.xticks(labels, [f"{(bin_width + i*bin_width):.2f}s" for i in range(len(labels)-1)] + ["No Coll. Soon"], fontsize=12, rotation=30)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x(), yval + 1, f"{yval:.2f}", size=9)

    print("="*7+"BIN ACCURACIES"+"="*7)
    for i, accuracy in enumerate(accuracy_per_label):
        if i == len(accuracy_per_label) - 1:
            print(f"No Coll. Soon: {accuracy:.2f}%")
        else:
            print(f"Coll. within {(i+1)*bin_width}s: {accuracy:.2f}%") if bin_width % 1 == 0 else print(f"Coll. within {(i+1)*bin_width:.2f}s: {accuracy:.2f}%")
    print("="*28)
    plt.tight_layout()

    if logger is not None:
        # Log to wandb through the logger
        logger.experiment.log({
            f"{log_prefix}_bin_accuracy": wandb.Image(fig, caption="Mean Accuracy per label")
        })

def log_image(
        image,
        log_prefix: str,
        logger
):
    """
    Log a single image to wandb.
    
    Args:
        image_tensor: Tensor of shape (C, T, H, W) or (C, H, W)
        log_prefix: Prefix for logging (e.g., "train", "val")
        logger: PyTorch Lightning logger (e.g., self.trainer.logger)
    """
    # take first image 
    image = image[:,0]  # (C, H, W)
    inverse_normalize = transforms.Normalize(
        mean=[-(0.485/0.229), -(0.456/0.224), -(0.406/0.225)],
        std=[1/0.229, 1/0.224, 1/0.225]
    )

    image = inverse_normalize(image).cpu().permute(1, 2, 0).float().numpy()  # (H, W, C)
    fig, axes = plt.subplots(
        1, 1, figsize=(10,
                        10)
    )
    axes.imshow(image)
    axes.axis("off")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(
        buf, format="png", bbox_inches="tight", pad_inches=0, facecolor="black"
    )
    plt.close(fig)
    buf.seek(0)
    PIL_image = Image.open(buf).convert('RGB')
    logger.experiment.log(  # type: ignore
        {
            f"{log_prefix}": [
                wandb.Image(PIL_image, file_type="jpg")
            ]
        }
        )
