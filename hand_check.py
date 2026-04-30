"""
Hand diagnostic for igris_c_v2_with_hands.xml

Run this to see:
  1. Did the finger meshes load? (or are they missing?)
  2. Where exactly are the fingertips in the world?
  3. Are they where you'd expect? (hands hanging down -> fingertips around hip level)
"""

import mujoco
import numpy as np

XML_PATH = "/home/home/mujoco_ws/src/igris_c_description/igris_c_v2_hand.xml"

try:
    model = mujoco.MjModel.from_xml_path(XML_PATH)
except Exception as e:
    print(f"!!! FAILED TO LOAD XML: {e}")
    print("   This likely means a mesh file is missing under <meshdir>/hand/")
    print("   Check the error message above for the missing filename.")
    raise

data = mujoco.MjData(model)

print(f"Model loaded ✓")
print(f"  nbody = {model.nbody}")
print(f"  nq    = {model.nq}  (qpos size)")
print(f"  nu    = {model.nu}  (ctrl size)")
print()

# Step once so kinematics are computed
mujoco.mj_forward(model, data)

# ---- Check if finger meshes loaded ----
print("=== Finger mesh check ===")
finger_mesh_names = [
    "Thumb_Proximal_Left", "Thumb_Middle_Left", "Thumb_Distal_Left",
    "Thumb_Proximal_Right", "Thumb_Middle_Right", "Thumb_Distal_Right",
    "Finger_Middle", "Finger_Distal",
]
for mname in finger_mesh_names:
    mid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, mname)
    if mid < 0:
        print(f"  {mname:25s}: NOT FOUND in compiled model")
    else:
        # mesh_vertnum = model.mesh_vertnum[mid]
        nv = model.mesh_vertnum[mid]
        nf = model.mesh_facenum[mid]
        print(f"  {mname:25s}: {nv} vertices, {nf} faces  (loaded)")
print()

# ---- Where are the fingertips in the world ? ----
print("=== Body world positions ===")
bodies_to_check = [
    "Left_Hand", "Right_Hand",
    "Left_Link_Thumb_Distal",  "Right_Link_Thumb_Distal",
    "Left_Link_Index_Distal",  "Right_Link_Index_Distal",
    "Left_Link_Middle_Distal", "Right_Link_Middle_Distal",
    "Left_Link_Ring_Distal",   "Right_Link_Ring_Distal",
    "Left_Link_Little_Distal", "Right_Link_Little_Distal",
]
for bname in bodies_to_check:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
    if bid < 0:
        print(f"  {bname:30s}: NOT FOUND")
        continue
    p = data.xpos[bid]
    print(f"  {bname:30s}: x={p[0]:+.3f}  y={p[1]:+.3f}  z={p[2]:+.3f}")

print()
print("=== Sanity check ===")
# In default pose, base_link is at (0, 0, 1.0). Arms hang down.
# Hands should be roughly at z ~ 0.4-0.6 (waist/hip level)
# Fingertips should be slightly below hands.
lh = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Left_Hand")]
li = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Left_Link_Index_Distal")]
print(f"Left_Hand z = {lh[2]:.3f}  (expected ~0.5 if arm hangs)")
print(f"Index fingertip z = {li[2]:.3f}")
print(f"Index fingertip y = {li[1]:.3f}  (should be POSITIVE for left side, around 0.15-0.25)")
print(f"Index fingertip x = {li[0]:.3f}  (should be near 0 if arm is straight down)")
print()
print(f"Vertical offset (fingertip below hand?): {li[2] - lh[2]:.3f}")
print(f"  - if NEGATIVE: fingertip is below hand  -> CORRECT (fingers point down with arm)")
print(f"  - if POSITIVE: fingertip is above hand  -> finger orientation is FLIPPED")