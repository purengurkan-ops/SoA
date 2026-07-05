import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# --- General settings ---
BG_COLOR = '#616161'
FG_COLOR = 'white'
TEXT_COLOR = 'black'

def draw_screen(ax, x, y_bottom, width, height, title, time_text):
    """Draws a screen with a title and time text."""
    rect = patches.Rectangle((x, y_bottom), width, height, linewidth=2, edgecolor='black', facecolor=BG_COLOR)
    ax.add_patch(rect)
    ax.text(x + width/2, y_bottom + height + 0.2, title, ha='center', va='bottom', fontsize=11, fontweight='bold', color=TEXT_COLOR)
    ax.text(x + width/2, y_bottom - 0.2, time_text, ha='center', va='top', fontsize=10, fontstyle='italic', color=TEXT_COLOR)

def draw_arrow(ax, x_start, x_end, y):
    """Draws a straight arrow between screens."""
    ax.annotate('', xy=(x_end, y), xytext=(x_start, y), arrowprops=dict(arrowstyle='->', lw=2, color='black'))

def draw_real_image(ax, x, y, image_path, zoom_level=0.06):
    """Draws real images without frames."""
    try:
        img = mpimg.imread(image_path)
        imagebox = OffsetImage(img, zoom=zoom_level)
        ab = AnnotationBbox(imagebox, (x, y), frameon=False)
        ax.add_artist(ab)
    except FileNotFoundError:
        ax.text(x, y, "IMG", color='red', ha='center', va='center', fontsize=8)

def draw_shape(ax, x, y, shape_type):
    """Draws simple shapes for calibration."""
    if shape_type == "square":
        patch = patches.Rectangle((x-0.2, y-0.2), 0.4, 0.4, color='black')
    else:
        patch = patches.Circle((x, y), 0.2, color='black')
    ax.add_patch(patch)

# --- FIGURE setup (3 Rows, 1 Column) ---
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 11))
fig.subplots_adjust(hspace=0.6) # Gap between panels

Y_BOT = 0.7
H = 2.9
Y_MID = Y_BOT + H/2
W = 3.0 # Screen width
GAP = 0.8 # Gap between screens

# ==========================================
# PANEL A: CALIBRATION
# ==========================================
ax0 = axes[0]
ax0.axis('off'); ax0.set_xlim(-0.5, 15); ax0.set_ylim(0, 4)
ax0.text(-0.5, 4.2, "A. Calibration Phase (Individual Thresholding)", fontsize=13, fontweight='bold', ha='left')

x = 0
draw_screen(ax0, x, Y_BOT, W, H, "Fixation", "500-800 ms")
ax0.text(x + W/2, Y_MID, "+", ha='center', va='center', fontsize=35, color=FG_COLOR)
draw_arrow(ax0, x + W + 0.1, x + W + GAP - 0.1, Y_MID)

x += W + GAP
draw_screen(ax0, x, Y_BOT, W, H, "Continuous Motion", "3000 ms")
draw_shape(ax0, x + 0.8, Y_MID, "square")
draw_shape(ax0, x + 2.0, Y_MID, "circle")
ax0.annotate('', xy=(x + 1.1, Y_MID + 0.5), xytext=(x + 0.8, Y_MID), arrowprops=dict(arrowstyle='->', color='white', lw=1.5, ls='--', connectionstyle="arc3,rad=-0.5"))
ax0.annotate('', xy=(x + 1.7, Y_MID - 0.5), xytext=(x + 2.0, Y_MID), arrowprops=dict(arrowstyle='->', color='white', lw=1.5, ls='--', connectionstyle="arc3,rad=0.5"))
draw_arrow(ax0, x + W + 0.1, x + W + GAP - 0.1, Y_MID)

x += W + GAP
draw_screen(ax0, x, Y_BOT, W, H, "Control Detection", "Max 3500 ms")
draw_shape(ax0, x + 0.8, Y_MID + 0.2, "square")
draw_shape(ax0, x + 2.0, Y_MID + 0.2, "circle")
ax0.text(x + W/2, Y_MID - 0.5, "Which shape did you control?", ha='center', fontsize=8.5, color=FG_COLOR, fontweight='bold')
ax0.text(x + 0.8, Y_MID - 0.9, "[A] Left", ha='center', fontsize=8, color=FG_COLOR)
ax0.text(x + 2.0, Y_MID - 0.9, "[S] Right", ha='center', fontsize=8, color=FG_COLOR)
draw_arrow(ax0, x + W + 0.1, x + W + GAP - 0.1, Y_MID)

x += W + GAP
draw_screen(ax0, x, Y_BOT, W, H, "Feedback", "800 ms")
ax0.text(x + W/2, Y_MID, "Right/Wrong", ha='center', va='center', fontsize=13, color=FG_COLOR, fontweight='bold')


# ==========================================
# PANEL B: ENCODING
# ==========================================
ax1 = axes[1]
ax1.axis('off'); ax1.set_xlim(-0.5, 15); ax1.set_ylim(0, 4)
ax1.text(-0.5, 4.2, "B. Encoding Phase (6 alternating 20-trial miniblocks)", fontsize=13, fontweight='bold', ha='left')

x = 0
draw_screen(ax1, x, Y_BOT, W, H, "Fixation", "500-800 ms")
ax1.text(x + W/2, Y_MID, "+", ha='center', va='center', fontsize=35, color=FG_COLOR)
draw_arrow(ax1, x + W + 0.1, x + W + GAP - 0.1, Y_MID)

x += W + GAP
draw_screen(ax1, x, Y_BOT, W, H, "Continuous Motion", "3000 ms")
draw_real_image(ax1, x + 0.8, Y_MID, "mouse2_11s.jpg", zoom_level=0.055)
draw_real_image(ax1, x + 2.0, Y_MID, "bean_04s.jpg", zoom_level=0.045)
ax1.annotate('', xy=(x + 1.1, Y_MID + 0.5), xytext=(x + 0.8, Y_MID), arrowprops=dict(arrowstyle='->', color='white', lw=1.5, ls='--', connectionstyle="arc3,rad=-0.5"))
ax1.annotate('', xy=(x + 1.7, Y_MID - 0.5), xytext=(x + 2.0, Y_MID), arrowprops=dict(arrowstyle='->', color='white', lw=1.5, ls='--', connectionstyle="arc3,rad=0.5"))
draw_arrow(ax1, x + W + 0.1, x + W + GAP - 0.1, Y_MID)

x += W + GAP
draw_screen(ax1, x, Y_BOT, W, H, "Control Detection", "Max 3500 ms")
# Reset to starting positions emphasize: images are shown again for control detection
draw_real_image(ax1, x + 0.8, Y_MID + 0.2, "mouse2_11s.jpg", zoom_level=0.055)
draw_real_image(ax1, x + 2.0, Y_MID + 0.2, "bean_04s.jpg", zoom_level=0.045)
ax1.text(x + W/2, Y_MID - 0.5, "Which image did you control?", ha='center', fontsize=9, color=FG_COLOR, fontweight='bold')
ax1.text(x + 0.8, Y_MID - 0.9, "[A] Left", ha='center', fontsize=8, color=FG_COLOR)
ax1.text(x + 2.0, Y_MID - 0.9, "[S] Right", ha='center', fontsize=8, color=FG_COLOR)
draw_arrow(ax1, x + W + 0.1, x + W + GAP - 0.1, Y_MID)

x += W + GAP
draw_screen(ax1, x, Y_BOT, W, H, "Agency Rating", "Self-paced")
ax1.text(x + W/2, Y_MID + 0.4, "How much control did you feel\nover the shape's movement?", ha='center', fontsize=8.5, color=FG_COLOR, fontweight='bold')
ax1.text(x + W/2, Y_MID - 0.2, "1 --------- 4 --------- 7", ha='center', fontsize=10, color=FG_COLOR)
ax1.text(x + W/2, Y_MID - 0.6, "Very weak               Very strong", ha='center', fontsize=7.5, color=FG_COLOR)


# ==========================================
# PANEL C: MEMORY TEST
# ==========================================
ax2 = axes[2]
ax2.axis('off'); ax2.set_xlim(-0.5, 15); ax2.set_ylim(0, 4)
ax2.text(-0.5, 4.2, "C. Surprise Recognition Memory Test", fontsize=13, fontweight='bold', ha='left')

x = 0
draw_screen(ax2, x, Y_BOT, W, H, "Fixation", "500-800 ms")
ax2.text(x + W/2, Y_MID, "+", ha='center', va='center', fontsize=35, color=FG_COLOR)
draw_arrow(ax2, x + W + 0.1, x + W + GAP - 0.1, Y_MID)

x += W + GAP
W_MEM = 4.0 
draw_screen(ax2, x, Y_BOT, W_MEM, H, "Recognition", "Self-paced")
draw_real_image(ax2, x + W_MEM/2, Y_MID + 0.4, "mouse2_11s.jpg", zoom_level=0.07)
ax2.text(x + W_MEM/2, Y_MID - 0.4, "Have you seen this image\nduring the experiment before?", ha='center', fontsize=10, color=FG_COLOR, fontweight='bold')
ax2.text(x + 1.0, Y_MID - 0.9, "Y (Yes)", ha='center', fontsize=9, color=FG_COLOR)
ax2.text(x + W_MEM - 1.0, Y_MID - 0.9, "N (No)", ha='center', fontsize=9, color=FG_COLOR)

# Save the final figure
plt.savefig("paradigm_complete_final.png", dpi=300, bbox_inches='tight')
print("Text details integrated and final figure created: paradigm_complete_final.png")