import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def arrange_figures(a_prefix, b_prefix, num_figures, output_filename="arranged_figures_ML.png", dpi=500):
    fig_width = 2.0  # Adjusting the figure width
    fig_height = 0.6 * num_figures
    fig, axes = plt.subplots(num_figures, 2, figsize=(fig_width, fig_height), gridspec_kw={"hspace": 0.001, "wspace": 0.001})

    for i in range(1, num_figures + 1):
        a_path = f"{a_prefix}_{i * 20}.png"
        b_path = f"{b_prefix}_{i * 20}.png"

        a_img = mpimg.imread(a_path)
        b_img = mpimg.imread(b_path)

        axes[i-1, 0].imshow(a_img)
        axes[i-1, 1].imshow(b_img)

        axes[i-1, 0].axis('off')
        axes[i-1, 1].axis('off')

    plt.savefig(output_filename, bbox_inches="tight", dpi=dpi)
    plt.show()

arrange_figures("Random_ML", "Research_ML", 5, dpi=700)  # You can adjust the number of figures (5 in this case)