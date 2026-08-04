import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# --- General settings ---
BG_COLOR = '#616161'      # grey "screen" background
FG_COLOR = 'white'        # on-screen text / stimuli
TEXT_COLOR = 'black'      # figure labels (titles, timings)

# --- Uniform screen geometry (SAME shape for every box = the monitor) -----
# Boxes are enlarged to use the empty space between phases; images scale up too.
W   = 3.5    # screen width  (identical across ALL panels, incl. recognition)
H   = 3.3    # screen height (identical across ALL panels)
GAP = 0.45   # horizontal gap between screens (arrows)
Y_BOT = 0.75
Y_MID = Y_BOT + H / 2
TOP   = Y_BOT + H

# stimulus image zoom levels (bigger boxes -> bigger images)
ZOOM_MOUSE = 0.072
ZOOM_BEAN  = 0.058
ZOOM_REC   = 0.075    # single recognition image (no horizontal neighbour)

# horizontal offset from box centre for the two side-by-side stimuli
DX = 0.9


def draw_screen(ax, x, title, time_text):
    """Draws one grey 'screen' with a bold title above and a timing below."""
    rect = patches.Rectangle((x, Y_BOT), W, H, linewidth=2,
                             edgecolor='black', facecolor=BG_COLOR)
    ax.add_patch(rect)
    ax.text(x + W / 2, TOP + 0.14, title, ha='center', va='bottom',
            fontsize=11, fontweight='bold', color=TEXT_COLOR)
    ax.text(x + W / 2, Y_BOT - 0.16, time_text, ha='center', va='top',
            fontsize=10, fontstyle='italic', color=TEXT_COLOR)


def draw_arrow(ax, x_screen_right):
    """Straight arrow from the right edge of one screen to the next."""
    ax.annotate('', xy=(x_screen_right + GAP - 0.06, Y_MID),
                xytext=(x_screen_right + 0.06, Y_MID),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))


def draw_real_image(ax, x, y, image_path, zoom_level):
    """Draws real stimulus images without frames (placeholder if missing)."""
    try:
        img = mpimg.imread(image_path)
        imagebox = OffsetImage(img, zoom=zoom_level)
        ab = AnnotationBbox(imagebox, (x, y), frameon=False)
        ax.add_artist(ab)
    except FileNotFoundError:
        ax.text(x, y, "IMG", color='red', ha='center', va='center', fontsize=9)


def draw_shape(ax, x, y, shape_type):
    """Draws the simple calibration shapes (square + circle)."""
    if shape_type == "square":
        patch = patches.Rectangle((x - 0.25, y - 0.25), 0.5, 0.5, color='black')
    else:
        patch = patches.Circle((x, y), 0.25, color='black')
    ax.add_patch(patch)


def motion_arrows(ax, cx):
    """The two dashed 'movement' arrows on the continuous-motion screens."""
    ax.annotate('', xy=(cx - 0.55, Y_MID + 0.6), xytext=(cx - 0.9, Y_MID),
                arrowprops=dict(arrowstyle='->', color='white', lw=1.5, ls='--',
                                connectionstyle="arc3,rad=-0.5"))
    ax.annotate('', xy=(cx + 0.55, Y_MID - 0.6), xytext=(cx + 0.9, Y_MID),
                arrowprops=dict(arrowstyle='->', color='white', lw=1.5, ls='--',
                                connectionstyle="arc3,rad=0.5"))


# --- FIGURE setup (3 Rows, 1 Column) ---
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 12))
fig.subplots_adjust(hspace=0.35, top=0.97, bottom=0.03)

for ax in axes:
    ax.axis('off')
    ax.set_xlim(-0.5, 15.4)
    ax.set_ylim(0, 4.9)

# ==========================================
# PANEL A: CALIBRATION
# ==========================================
ax0 = axes[0]
ax0.text(-0.5, 4.6, "A. Calibration Phase (Individual Thresholding)",
         fontsize=13, fontweight='bold', ha='left')

x = 0
cx = x + W / 2
draw_screen(ax0, x, "Fixation", "500-800 ms")
ax0.text(cx, Y_MID, "+", ha='center', va='center', fontsize=40, color=FG_COLOR)
draw_arrow(ax0, x + W)

x += W + GAP
cx = x + W / 2
draw_screen(ax0, x, "Continuous Motion", "3000 ms")
draw_shape(ax0, cx - DX, Y_MID, "square")
draw_shape(ax0, cx + DX, Y_MID, "circle")
motion_arrows(ax0, cx)
draw_arrow(ax0, x + W)

x += W + GAP
cx = x + W / 2
draw_screen(ax0, x, "Control Detection", "Max 3500 ms")
draw_shape(ax0, cx - DX, Y_MID + 0.45, "square")
draw_shape(ax0, cx + DX, Y_MID + 0.45, "circle")
ax0.text(cx, Y_MID - 0.35, "Which shape did you control?",
         ha='center', fontsize=9, color=FG_COLOR, fontweight='bold')
ax0.text(cx - DX, Y_MID - 0.85, "A", ha='center', fontsize=12, color=FG_COLOR, fontweight='bold')
ax0.text(cx + DX, Y_MID - 0.85, "S", ha='center', fontsize=12, color=FG_COLOR, fontweight='bold')
draw_arrow(ax0, x + W)

x += W + GAP
cx = x + W / 2
draw_screen(ax0, x, "Feedback", "800 ms")
ax0.text(cx, Y_MID, "Right", ha='center', va='center',
         fontsize=16, color=FG_COLOR, fontweight='bold')


# ==========================================
# PANEL B: ENCODING
# ==========================================
ax1 = axes[1]
ax1.text(-0.5, 4.6, "B. Encoding Phase (6 alternating 20-trial miniblocks)",
         fontsize=13, fontweight='bold', ha='left')

x = 0
cx = x + W / 2
draw_screen(ax1, x, "Fixation", "500-800 ms")
ax1.text(cx, Y_MID, "+", ha='center', va='center', fontsize=40, color=FG_COLOR)
draw_arrow(ax1, x + W)

x += W + GAP
cx = x + W / 2
draw_screen(ax1, x, "Continuous Motion", "3000 ms")
draw_real_image(ax1, cx - DX, Y_MID, "mouse2_11s.jpg", ZOOM_MOUSE)
draw_real_image(ax1, cx + DX, Y_MID, "bean_04s.jpg", ZOOM_BEAN)
motion_arrows(ax1, cx)
draw_arrow(ax1, x + W)

x += W + GAP
cx = x + W / 2
draw_screen(ax1, x, "Control Detection", "Max 3500 ms")
draw_real_image(ax1, cx - DX, Y_MID + 0.45, "mouse2_11s.jpg", ZOOM_MOUSE)
draw_real_image(ax1, cx + DX, Y_MID + 0.45, "bean_04s.jpg", ZOOM_BEAN)
ax1.text(cx, Y_MID - 0.55, "Which image did you control?",
         ha='center', fontsize=9, color=FG_COLOR, fontweight='bold')
ax1.text(cx - DX, Y_MID - 1.0, "A", ha='center', fontsize=12, color=FG_COLOR, fontweight='bold')
ax1.text(cx + DX, Y_MID - 1.0, "S", ha='center', fontsize=12, color=FG_COLOR, fontweight='bold')
draw_arrow(ax1, x + W)

x += W + GAP
cx = x + W / 2
draw_screen(ax1, x, "Agency Rating", "Self-paced")
ax1.text(cx, Y_MID + 0.7, "How much control did you feel\nover the shape's movement?",
         ha='center', fontsize=9, color=FG_COLOR, fontweight='bold')
ax1.text(cx, Y_MID - 0.05, "1    2    3    4    5    6    7",
         ha='center', fontsize=11, color=FG_COLOR)
ax1.text(cx - 1.15, Y_MID - 0.55, "Very weak", ha='center', fontsize=7.5, color=FG_COLOR)
ax1.text(cx, Y_MID - 0.55, "Moderate", ha='center', fontsize=7.5, color=FG_COLOR)
ax1.text(cx + 1.15, Y_MID - 0.55, "Very strong", ha='center', fontsize=7.5, color=FG_COLOR)


# ==========================================
# PANEL C: MEMORY TEST
# ==========================================
ax2 = axes[2]
ax2.text(-0.5, 4.6, "C. Surprise Recognition Memory Test",
         fontsize=13, fontweight='bold', ha='left')

x = 0
cx = x + W / 2
draw_screen(ax2, x, "Fixation", "500-800 ms")
ax2.text(cx, Y_MID, "+", ha='center', va='center', fontsize=40, color=FG_COLOR)
draw_arrow(ax2, x + W)

x += W + GAP
cx = x + W / 2
draw_screen(ax2, x, "Recognition", "Self-paced")   # SAME size box as all others
draw_real_image(ax2, cx, Y_MID + 0.55, "mouse2_11s.jpg", ZOOM_REC)
ax2.text(cx, Y_MID - 0.55, "Have you seen this image\nduring the experiment before?",
         ha='center', fontsize=9, color=FG_COLOR, fontweight='bold')
ax2.text(cx - DX, Y_MID - 1.15, "Y\nYes", ha='center', fontsize=10, color=FG_COLOR, fontweight='bold')
ax2.text(cx + DX, Y_MID - 1.15, "N\nNo", ha='center', fontsize=10, color=FG_COLOR, fontweight='bold')

# Save the final figure
plt.savefig("paradigm_complete_final.png", dpi=300, bbox_inches='tight')
print("Figure saved: paradigm_complete_final.png")