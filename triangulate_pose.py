"""
Triangulate 3D skeleton pose from two adjacent cameras and compare with ground truth.

This script:
1. Loads 2D keypoints from two cameras (blu79CF and grn43E3 stereo pair)
2. Triangulates 3D positions from the 2D points using stereo geometry
3. Extracts/estimates 3D ground truth from SMPL mesh
4. Plots both triangulated and ground truth skeletons in 3D
"""

import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from plyfile import PlyData

# Skeleton connection definition for visualization
# Maps each joint to its parent for drawing bones
SKELETON_CONNECTIONS = [
    # Head and face
    ('head', 'neck'),
    ('head', 'nose'),
    ('nose', 'reye'),
    ('nose', 'leye'),
    # Upper body
    ('neck', 'rsho'),
    ('neck', 'lsho'),
    ('rsho', 'relb'),
    ('lsho', 'lelb'),
    ('relb', 'rwri'),
    ('lelb', 'lwri'),
    # Spine to hips
    ('neck', 'rhip'),
    ('neck', 'lhip'),
    ('rhip', 'lhip'),  # Pelvis connection
    # Legs
    ('rhip', 'rknee'),
    ('lhip', 'lknee'),
    ('rknee', 'rankl'),
    ('lknee', 'lankl'),
]

# SMPL joint indices for approximating keypoints from mesh vertices
# These are standard SMPL vertex indices that correspond to body parts
# SMPL has 6890 vertices with a well-defined topology
SMPL_KEYPOINT_VERTEX_INDICES = {
    'head': 411,          # Head top
    'neck': 3068,         # Neck
    'nose': 332,          # Nose tip
    'reye': 6260,         # Right eye area
    'leye': 2800,         # Left eye area  
    'mouth': 3506,        # Mouth/chin area
    'rsho': 5282,         # Right shoulder
    'lsho': 1861,         # Left shoulder
    'relb': 5049,         # Right elbow
    'lelb': 1618,         # Left elbow
    'rwri': 5530,         # Right wrist
    'lwri': 2098,         # Left wrist
    'rhip': 6513,         # Right hip
    'lhip': 3121,         # Left hip
    'rknee': 4494,        # Right knee
    'lknee': 1013,        # Left knee
    'rankl': 6728,        # Right ankle
    'lankl': 3327,        # Left ankle
}


def load_2d_keypoints(basedir, capture_date, camera_name, frame_id, track_id):
    """Load 2D keypoints from a label file."""
    fn = os.path.join(basedir, f'{capture_date}_{camera_name}_{frame_id:07d}_{track_id}.json')
    if not os.path.exists(fn):
        print(f"File not found: {fn}")
        return None
    with open(fn, 'r') as f:
        data = json.load(f)
    return data['keypoint']


def load_3d_mesh(basedir, capture_date, frame_id, track_id):
    """Load 3D mesh from SMPL ply file."""
    fn = os.path.join(basedir, f'{capture_date}_{frame_id:07d}_{track_id}.ply')
    if not os.path.exists(fn):
        print(f"File not found: {fn}")
        return None
    plydata = PlyData.read(fn)
    x = plydata['vertex']['x']
    y = plydata['vertex']['y']
    z = plydata['vertex']['z']
    return np.column_stack([x, y, z])


def extract_keypoints_from_mesh(mesh_vertices):
    """Extract keypoint positions from SMPL mesh vertices."""
    keypoints_3d = {}
    for name, vertex_idx in SMPL_KEYPOINT_VERTEX_INDICES.items():
        if vertex_idx < len(mesh_vertices):
            keypoints_3d[name] = mesh_vertices[vertex_idx]
    return keypoints_3d


def get_camera_parameters():
    """
    Get camera intrinsic and extrinsic parameters.
    
    The PedX dataset uses blu79CF-grn43E3 as one stereo pair.
    Since actual calibration files are not available in the demo,
    we estimate reasonable parameters based on:
    - Image resolution (3645 x 2687 for blu/grn cameras)
    - Typical industrial stereo camera baselines (~10-20cm for close range)
    
    Note: These are estimated parameters. For accurate triangulation,
    use the actual calibration files from the full PedX dataset.
    """
    # Image dimensions for blu79CF and grn43E3
    img_width = 3645
    img_height = 2687
    
    # Estimated focal length in pixels (assuming ~50mm lens on APS-C sensor equivalent)
    # focal_length_px = focal_mm * sensor_width_pixels / sensor_width_mm
    # Assuming 35mm equiv of ~50mm, with 23.5mm sensor width
    focal_length_px = 3000  # pixels (estimated)
    
    # Principal point at image center
    cx = img_width / 2
    cy = img_height / 2
    
    # Camera intrinsic matrix (same for both cameras in rectified stereo)
    K = np.array([
        [focal_length_px, 0, cx],
        [0, focal_length_px, cy],
        [0, 0, 1]
    ])
    
    # Stereo baseline (distance between cameras)
    # Estimated at 10cm for typical stereo setup
    baseline = 0.1  # meters
    
    # For a rectified stereo pair:
    # Left camera (blu79CF) at origin
    R1 = np.eye(3)
    t1 = np.zeros(3)
    
    # Right camera (grn43E3) translated along X-axis
    R2 = np.eye(3)
    t2 = np.array([baseline, 0, 0])
    
    # Projection matrices
    P1 = K @ np.hstack([R1, t1.reshape(3, 1)])
    P2 = K @ np.hstack([R2, t2.reshape(3, 1)])
    
    return K, P1, P2, baseline


def triangulate_point(pt1, pt2, P1, P2):
    """
    Triangulate a 3D point from two 2D observations using DLT method.
    
    Args:
        pt1: 2D point in camera 1 (x, y)
        pt2: 2D point in camera 2 (x, y)
        P1: Projection matrix of camera 1 (3x4)
        P2: Projection matrix of camera 2 (3x4)
    
    Returns:
        3D point in world coordinates
    """
    # Build the linear system A * X = 0
    A = np.zeros((4, 4))
    A[0] = pt1[0] * P1[2] - P1[0]
    A[1] = pt1[1] * P1[2] - P1[1]
    A[2] = pt2[0] * P2[2] - P2[0]
    A[3] = pt2[1] * P2[2] - P2[1]
    
    # Solve using SVD
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    
    # Convert from homogeneous coordinates
    X = X[:3] / X[3]
    
    return X


def procrustes_alignment(source_points, target_points):
    """
    Align source points to target points using Procrustes analysis.
    Returns: translated, scaled, and rotated source points
    """
    # Center both point sets
    source_centroid = np.mean(source_points, axis=0)
    target_centroid = np.mean(target_points, axis=0)
    
    source_centered = source_points - source_centroid
    target_centered = target_points - target_centroid
    
    # Compute scale
    source_scale = np.sqrt(np.sum(source_centered ** 2))
    target_scale = np.sqrt(np.sum(target_centered ** 2))
    
    if source_scale < 1e-10:
        return source_points
    
    source_normalized = source_centered / source_scale
    target_normalized = target_centered / target_scale
    
    # Find optimal rotation using SVD
    H = source_normalized.T @ target_normalized
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Correct for reflection if needed
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Apply transformation
    scale = target_scale / source_scale
    aligned = (source_centered @ R) * scale + target_centroid
    
    return aligned


def triangulate_skeleton(keypoints_cam1, keypoints_cam2, P1, P2):
    """
    Triangulate all keypoints from two camera views.
    
    Args:
        keypoints_cam1: dict of keypoints from camera 1
        keypoints_cam2: dict of keypoints from camera 2
        P1, P2: Projection matrices
    
    Returns:
        dict of 3D keypoint positions
    """
    keypoints_3d = {}
    
    common_keypoints = set(keypoints_cam1.keys()) & set(keypoints_cam2.keys())
    
    for name in common_keypoints:
        kp1 = keypoints_cam1[name]
        kp2 = keypoints_cam2[name]
        
        # Only triangulate if both keypoints are visible
        if kp1.get('visible', True) and kp2.get('visible', True):
            pt1 = np.array([kp1['x'], kp1['y']])
            pt2 = np.array([kp2['x'], kp2['y']])
            
            keypoints_3d[name] = triangulate_point(pt1, pt2, P1, P2)
    
    return keypoints_3d


def plot_skeleton_3d(ax, keypoints_3d, color='b', label='', marker='o', linewidth=2):
    """Plot a 3D skeleton on the given axes."""
    # Plot joints
    for name, pos in keypoints_3d.items():
        ax.scatter(*pos, c=color, marker=marker, s=50)
    
    # Plot bones
    for joint1, joint2 in SKELETON_CONNECTIONS:
        if joint1 in keypoints_3d and joint2 in keypoints_3d:
            pos1 = keypoints_3d[joint1]
            pos2 = keypoints_3d[joint2]
            ax.plot([pos1[0], pos2[0]], 
                   [pos1[1], pos2[1]], 
                   [pos1[2], pos2[2]], 
                   c=color, linewidth=linewidth, label=label if joint1 == 'head' else '')


def visualize_triangulation(basedir_2d, basedir_3d, capture_date, frame_id, track_id, 
                           output_file=None):
    """
    Main visualization function that triangulates 2D points and compares with ground truth.
    
    Args:
        basedir_2d: Path to 2D labels directory
        basedir_3d: Path to 3D labels directory
        capture_date: Capture date string (e.g., '20171207T2024')
        frame_id: Frame number
        track_id: Tracking ID of the pedestrian
        output_file: Optional path to save the figure
    """
    # Load 2D keypoints from stereo pair
    keypoints_blu = load_2d_keypoints(basedir_2d, capture_date, 'blu79CF', frame_id, track_id)
    keypoints_grn = load_2d_keypoints(basedir_2d, capture_date, 'grn43E3', frame_id, track_id)
    
    if keypoints_blu is None or keypoints_grn is None:
        print("Failed to load 2D keypoints")
        return
    
    # Load 3D ground truth mesh
    mesh_3d = load_3d_mesh(basedir_3d, capture_date, frame_id, track_id)
    if mesh_3d is None:
        print("Failed to load 3D mesh")
        return
    
    # Extract ground truth keypoints from mesh
    gt_keypoints_3d = extract_keypoints_from_mesh(mesh_3d)
    
    # Get camera parameters and triangulate
    K, P1, P2, baseline = get_camera_parameters()
    triangulated_keypoints = triangulate_skeleton(keypoints_blu, keypoints_grn, P1, P2)
    
    # Store raw triangulated for display (centered at origin for visualization)
    raw_points = np.array(list(triangulated_keypoints.values()))
    raw_centroid = np.mean(raw_points, axis=0)
    raw_triangulated = {k: v - raw_centroid for k, v in triangulated_keypoints.items()}
    
    # Scale and transform triangulated points to match ground truth coordinate system
    # using Procrustes alignment for better results
    aligned_keypoints = {}
    if triangulated_keypoints and gt_keypoints_3d:
        # Get common keypoints
        common_keys = [k for k in triangulated_keypoints.keys() if k in gt_keypoints_3d]
        
        if len(common_keys) >= 4:
            tri_points = np.array([triangulated_keypoints[k] for k in common_keys])
            gt_points = np.array([gt_keypoints_3d[k] for k in common_keys])
            
            # Align using Procrustes
            aligned_points = procrustes_alignment(tri_points, gt_points)
            
            # Update aligned keypoints
            for i, name in enumerate(common_keys):
                aligned_keypoints[name] = aligned_points[i]
    
    # Create figure with 2x2 layout for better comparison
    fig = plt.figure(figsize=(14, 12))
    
    # Get axis limits from ground truth
    gt_points_arr = np.array(list(gt_keypoints_3d.values()))
    center = np.mean(gt_points_arr, axis=0)
    max_range = np.max(np.abs(gt_points_arr - center)) * 1.5
    
    # Plot 1: Ground truth skeleton
    ax1 = fig.add_subplot(221, projection='3d')
    plot_skeleton_3d(ax1, gt_keypoints_3d, color='green', label='Ground Truth', marker='o')
    ax1.set_title('Ground Truth\n(from SMPL mesh)', fontsize=11)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_xlim([center[0] - max_range, center[0] + max_range])
    ax1.set_ylim([center[1] - max_range, center[1] + max_range])
    ax1.set_zlim([center[2] - max_range, center[2] + max_range])
    
    # Plot 2: Raw triangulated skeleton (centered at origin)
    ax2 = fig.add_subplot(222, projection='3d')
    plot_skeleton_3d(ax2, raw_triangulated, color='red', label='Raw Triangulated', marker='s')
    ax2.set_title('Raw Triangulated\n(before alignment, centered)', fontsize=11)
    ax2.set_xlabel('X (relative)')
    ax2.set_ylabel('Y (relative)')
    ax2.set_zlabel('Z (relative)')
    
    # Plot 3: Aligned triangulated skeleton  
    ax3 = fig.add_subplot(223, projection='3d')
    plot_skeleton_3d(ax3, aligned_keypoints, color='blue', label='Aligned', marker='^')
    ax3.set_title('Triangulated\n(Procrustes aligned)', fontsize=11)
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_zlabel('Z (m)')
    ax3.set_xlim([center[0] - max_range, center[0] + max_range])
    ax3.set_ylim([center[1] - max_range, center[1] + max_range])
    ax3.set_zlim([center[2] - max_range, center[2] + max_range])
    
    # Plot 4: Overlay comparison
    ax4 = fig.add_subplot(224, projection='3d')
    plot_skeleton_3d(ax4, gt_keypoints_3d, color='green', label='Ground Truth', marker='o', linewidth=2)
    plot_skeleton_3d(ax4, aligned_keypoints, color='blue', label='Triangulated', marker='^', linewidth=1)
    ax4.set_title('Overlay Comparison', fontsize=11)
    ax4.set_xlabel('X (m)')
    ax4.set_ylabel('Y (m)')
    ax4.set_zlabel('Z (m)')
    ax4.set_xlim([center[0] - max_range, center[0] + max_range])
    ax4.set_ylim([center[1] - max_range, center[1] + max_range])
    ax4.set_zlim([center[2] - max_range, center[2] + max_range])
    ax4.legend(loc='upper left')
    
    plt.suptitle(f'3D Pose Triangulation vs Ground Truth\n'
                 f'Stereo pair: blu79CF - grn43E3 | Frame: {frame_id} | Track: {track_id[-8:]}', 
                 fontsize=12)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved figure to: {output_file}")
    
    plt.show()
    
    # Print summary statistics
    print("\n=== Triangulation Summary ===")
    print(f"Frame ID: {frame_id}")
    print(f"Track ID: {track_id}")
    print(f"Stereo pair: blu79CF - grn43E3")
    print(f"Common keypoints triangulated: {len(aligned_keypoints)}")
    
    if aligned_keypoints and gt_keypoints_3d:
        errors = []
        print("\nPer-joint errors (after Procrustes alignment):")
        for name in sorted(aligned_keypoints.keys()):
            if name in gt_keypoints_3d:
                error = np.linalg.norm(aligned_keypoints[name] - gt_keypoints_3d[name])
                errors.append(error)
                print(f"  {name:<8}: {error:.4f} m")
        
        if errors:
            print(f"\nMean error: {np.mean(errors):.4f} m")
            print(f"Max error:  {np.max(errors):.4f} m")
            print(f"Min error:  {np.min(errors):.4f} m")
            
    print("\nNote: Without actual camera calibration files, triangulation uses")
    print("      estimated parameters. For accurate results, use the calibration")
    print("      files from the full PedX dataset (calib_cam_to_cam_blu79CF-grn43E3.txt)")


def main():
    """Main entry point."""
    # Base directories for labels
    basedir_2d = './labels-2d_20171207T2024/20171207T2024'
    basedir_3d = './labels-3d-pose_20171207T2024/20171207T2024'
    
    # Demo parameters
    capture_date = '20171207T2024'
    frame_id = 55
    
    # Find available track IDs for this frame
    pattern = os.path.join(basedir_2d, f'{capture_date}_blu79CF_{frame_id:07d}_*.json')
    files = glob.glob(pattern)
    
    if not files:
        print(f"No labels found for frame {frame_id}")
        return
    
    # Extract track IDs
    track_ids = [os.path.basename(f).split('_')[-1].replace('.json', '') for f in files]
    print(f"Found {len(track_ids)} pedestrians in frame {frame_id}")
    print(f"Track IDs: {track_ids}")
    
    # Visualize all pedestrians
    for i, track_id in enumerate(track_ids):
        print(f"\n{'='*60}")
        print(f"Processing pedestrian {i+1}/{len(track_ids)}: {track_id[-8:]}")
        print('='*60)
        
        output_file = f'triangulation_result_{capture_date}_{frame_id:07d}_{track_id[-8:]}.png'
        visualize_triangulation(basedir_2d, basedir_3d, capture_date, frame_id, track_id, 
                               output_file=output_file)
    
    print(f"\n{'='*60}")
    print(f"Completed processing {len(track_ids)} pedestrians")
    print(f"Output files: triangulation_result_{capture_date}_{frame_id:07d}_*.png")
    print('='*60)


if __name__ == '__main__':
    main()
