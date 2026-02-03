"""
Triangulate 3D skeleton pose from two adjacent cameras and compare with ground truth.

This script:
1. Loads 2D keypoints from two cameras (blu79CF and grn43E3 stereo pair)
2. Uses calibration data from ./data/calib for proper camera parameters
3. Triangulates 3D positions from the 2D points using calibrated stereo geometry
4. Extracts/estimates 3D ground truth from SMPL mesh
5. Creates 3D scene visualizations for each frame with all pedestrians
6. Generates a video showing triangulated poses vs ground truth across all frames
"""

import os
import glob
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for video generation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from plyfile import PlyData
import cv2
from pedx.utils import read_calib_file

# Calibration file path (constant to avoid duplication)
CALIB_FILE = './data/calib/calib_cam_to_cam_blu79CF-grn43E3.txt'

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


def parse_label_filename(filename):
    """
    Parse a 2D label filename to extract frame_id and track_id.
    
    Expected filename format: {capture_date}_{camera_name}_{frame_id:07d}_{track_id}.json
    Example: 20171207T2024_blu79CF_0000055_15d1ab9781b84825b03a9cf713f52ed5.json
    
    Returns:
        tuple: (frame_id, track_id) or (None, None) if parsing fails
    """
    basename = os.path.basename(filename)
    name_without_ext = basename.replace('.json', '')
    parts = name_without_ext.split('_')
    
    if len(parts) >= 4:
        # frame_id is the 3rd part (index 2), track_id is the 4th part (index 3)
        try:
            frame_id = int(parts[2])
            track_id = parts[3]
            return frame_id, track_id
        except (ValueError, IndexError):
            return None, None
    return None, None


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
    """
    Extract keypoint positions from SMPL mesh vertices.
    
    The SMPL mesh in this dataset has coordinates where:
    - X is left-right
    - Y is forward-backward (roughly constant for upright pose)
    - Z points downward (head has lower Z than feet)
    
    We convert to a coordinate system where Z is up (standard for matplotlib 3D):
    - X stays the same (left-right)
    - Y stays the same (forward-backward/depth)
    - Z becomes the vertical axis (use -smpl_Z so head points up)
    """
    keypoints_3d = {}
    for name, vertex_idx in SMPL_KEYPOINT_VERTEX_INDICES.items():
        if vertex_idx < len(mesh_vertices):
            smpl_coords = mesh_vertices[vertex_idx]
            # Convert coordinates: negate Z to make head point upward (positive Z)
            keypoints_3d[name] = np.array([
                smpl_coords[0],    # X stays the same (left-right)
                smpl_coords[1],    # Y stays the same (forward-backward)
                -smpl_coords[2]    # -Z becomes Z (vertical up)
            ])
    return keypoints_3d


def get_camera_parameters():
    """
    Get camera intrinsic parameters from calibration file.
    
    The PedX dataset uses blu79CF-grn43E3 as one stereo pair.
    Returns intrinsic matrices K for both cameras and image dimensions.
    """
    # Load calibration from file
    calib = read_calib_file(CALIB_FILE)
    
    # Image dimensions from S_rect_00 (rectified image size)
    img_width = int(calib['S_rect_00'][0])
    img_height = int(calib['S_rect_00'][1])
    
    # Intrinsic matrix K from K_00 (camera 0 = blu79CF) - 3x3 stored row-major
    K_00 = calib['K_00'].reshape(3, 3)
    
    # Intrinsic matrix K from K_01 (camera 1 = grn43E3)
    K_01 = calib['K_01'].reshape(3, 3)
    
    # Distortion coefficients (5 elements)
    D_00 = calib['D_00']
    D_01 = calib['D_01']
    
    return K_00, K_01, D_00, D_01, img_width, img_height


def get_stereo_rectified_parameters():
    """
    Get stereo-rectified projection matrices from calibration file.
    
    Returns the rectified projection matrices P_rect_00 and P_rect_01
    which can be used directly for triangulation.
    """
    # Load calibration from file
    calib = read_calib_file(CALIB_FILE)
    
    # Rectified projection matrices (3x4)
    # P_rect_00 = [K | 0] for the reference camera
    # P_rect_01 = [K | -b*fx] where b is baseline
    P_rect_00 = calib['P_rect_00'].reshape(3, 4)
    P_rect_01 = calib['P_rect_01'].reshape(3, 4)
    
    # Rectification rotation matrices
    R_rect_00 = calib['R_rect_00'].reshape(3, 3)
    R_rect_01 = calib['R_rect_01'].reshape(3, 3)
    
    # Q matrix for reprojection from disparity (4x4)
    Q = calib['Q'].reshape(4, 4)
    
    return P_rect_00, P_rect_01, R_rect_00, R_rect_01, Q


def estimate_camera_params_pnp(keypoints_2d, keypoints_3d, K):
    """
    Estimate camera extrinsic parameters using PnP (Perspective-n-Point).
    
    Given known 2D keypoints and corresponding 3D ground truth positions,
    we can solve for the camera's rotation and translation.
    
    Args:
        keypoints_2d: dict of 2D keypoints from the camera
        keypoints_3d: dict of 3D ground truth keypoint positions
        K: Camera intrinsic matrix (3x3)
    
    Returns:
        R: Rotation matrix (3x3)
        t: Translation vector (3,)
        P: Projection matrix (3x4)
        success: Whether PnP succeeded
    """
    # Get common keypoints between 2D and 3D
    common_keys = []
    for name in keypoints_2d.keys():
        if name in keypoints_3d and keypoints_2d[name].get('visible', True):
            common_keys.append(name)
    
    if len(common_keys) < 4:
        # PnP requires at least 4 point correspondences to uniquely solve for the 6 DoF camera pose
        print(f"Warning: Not enough common keypoints for PnP ({len(common_keys)} < 4 required)")
        return None, None, None, False
    
    # Build arrays of 2D and 3D points
    points_2d = np.array([[keypoints_2d[k]['x'], keypoints_2d[k]['y']] 
                          for k in common_keys], dtype=np.float64)
    points_3d = np.array([keypoints_3d[k] for k in common_keys], dtype=np.float64)
    
    # Solve PnP using RANSAC for robustness
    dist_coeffs = np.zeros(4)  # Assuming no lens distortion
    # Reprojection error threshold: 8 pixels is reasonable for high-res images (3645x2687)
    # This allows for some annotation noise while rejecting outliers
    reprojection_threshold = 8.0
    
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        points_3d, points_2d, K, dist_coeffs,
        iterationsCount=1000,
        reprojectionError=reprojection_threshold,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    if not success:
        print("Warning: PnP failed to find a solution")
        return None, None, None, False
    
    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.flatten()
    
    # Build projection matrix P = K * [R | t]
    P = K @ np.hstack([R, t.reshape(3, 1)])
    
    return R, t, P, True


def estimate_averaged_camera_params(basedir_2d, basedir_3d, capture_date, frame_id, 
                                     camera_name, track_ids, K):
    """
    Estimate camera parameters averaged over multiple pedestrians.
    
    This provides more robust camera parameter estimates by using multiple
    2D-3D correspondences from different pedestrians in the same frame.
    
    Args:
        basedir_2d: Path to 2D labels directory
        basedir_3d: Path to 3D labels directory
        capture_date: Capture date string
        frame_id: Frame number
        camera_name: Camera name (e.g., 'blu79CF')
        track_ids: List of pedestrian tracking IDs
        K: Camera intrinsic matrix (3x3)
    
    Returns:
        R: Averaged rotation matrix (3x3)
        t: Averaged translation vector (3,)
        P: Projection matrix from averaged params (3x4)
        success: Whether averaging succeeded
    """
    rotation_vecs = []
    translations = []
    
    for track_id in track_ids:
        # Load 2D keypoints for this pedestrian
        keypoints_2d = load_2d_keypoints(basedir_2d, capture_date, camera_name, frame_id, track_id)
        if keypoints_2d is None:
            continue
        
        # Load 3D ground truth mesh
        mesh_3d = load_3d_mesh(basedir_3d, capture_date, frame_id, track_id)
        if mesh_3d is None:
            continue
        
        # Extract ground truth keypoints
        gt_keypoints_3d = extract_keypoints_from_mesh(mesh_3d)
        
        # Estimate camera params for this pedestrian
        R, t, P, success = estimate_camera_params_pnp(keypoints_2d, gt_keypoints_3d, K)
        
        if success:
            # Store rotation as rotation vector for averaging
            rvec, _ = cv2.Rodrigues(R)
            rotation_vecs.append(rvec.flatten())
            translations.append(t)
    
    if len(rotation_vecs) < 1:
        print(f"Warning: No successful PnP estimates for camera {camera_name}")
        return None, None, None, False
    
    # Average the rotation vectors and translations
    # Note: Averaging rotation vectors is an approximation that works well when rotations
    # are similar (as expected for cameras viewing the same scene). For significantly
    # different rotations, more sophisticated methods like quaternion averaging or
    # computing the Karcher mean on SO(3) would be more accurate.
    avg_rvec = np.mean(rotation_vecs, axis=0)
    avg_t = np.mean(translations, axis=0)
    
    # Convert averaged rotation vector back to rotation matrix
    avg_R, _ = cv2.Rodrigues(avg_rvec)
    
    # Build projection matrix
    avg_P = K @ np.hstack([avg_R, avg_t.reshape(3, 1)])
    
    print(f"Camera {camera_name}: averaged params from {len(rotation_vecs)} pedestrians")
    
    return avg_R, avg_t, avg_P, True


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
                           output_file=None, P1=None, P2=None):
    """
    Main visualization function that triangulates 2D points and compares with ground truth.
    
    Uses PnP (Perspective-n-Point) to estimate camera parameters from known 2D-3D 
    correspondences, then performs triangulation with the estimated cameras.
    
    Args:
        basedir_2d: Path to 2D labels directory
        basedir_3d: Path to 3D labels directory
        capture_date: Capture date string (e.g., '20171207T2024')
        frame_id: Frame number
        track_id: Tracking ID of the pedestrian
        output_file: Optional path to save the figure
        P1: Optional pre-computed projection matrix for camera 1 (blu79CF)
        P2: Optional pre-computed projection matrix for camera 2 (grn43E3)
    """
    # Track if we're using averaged (pre-computed) parameters
    using_averaged_params = P1 is not None and P2 is not None
    
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
    
    # Extract ground truth keypoints from mesh (with Y-up to Z-up conversion)
    gt_keypoints_3d = extract_keypoints_from_mesh(mesh_3d)
    
    # Get camera intrinsic parameters
    K, img_width, img_height = get_camera_parameters()
    
    # Use provided projection matrices or estimate them
    if P1 is None or P2 is None:
        # Estimate camera extrinsics using PnP from 2D-3D correspondences.
        # NOTE: This approach uses GT 3D keypoints to estimate camera parameters, which
        # creates a circular dependency. This is intentional for validation/testing purposes
        # to demonstrate triangulation quality. In production without GT, you would need
        # actual camera calibration files (e.g., calib_cam_to_cam_blu79CF-grn43E3.txt).
        R1, t1, P1, success1 = estimate_camera_params_pnp(keypoints_blu, gt_keypoints_3d, K)
        R2, t2, P2, success2 = estimate_camera_params_pnp(keypoints_grn, gt_keypoints_3d, K)
        
        if not success1 or not success2:
            print("Warning: PnP estimation failed for one or both cameras")
            print("Triangulation may be inaccurate")
            # Fallback to simple stereo configuration
            R1, t1 = np.eye(3), np.zeros(3)
            R2, t2 = np.eye(3), np.array([0.1, 0, 0])
            P1 = K @ np.hstack([R1, t1.reshape(3, 1)])
            P2 = K @ np.hstack([R2, t2.reshape(3, 1)])
    
    # Triangulate using camera parameters
    triangulated_keypoints = triangulate_skeleton(keypoints_blu, keypoints_grn, P1, P2)
    
    # Store raw triangulated for display (centered at origin for visualization)
    if triangulated_keypoints:
        raw_points = np.array(list(triangulated_keypoints.values()))
        raw_centroid = np.mean(raw_points, axis=0)
        raw_triangulated = {k: v - raw_centroid for k, v in triangulated_keypoints.items()}
    else:
        raw_triangulated = {}
    
    # Scale and transform triangulated points to match ground truth coordinate system
    # using Procrustes alignment for comparison
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
    ax2.set_title('Raw Triangulated\n(PnP-based, centered)', fontsize=11)
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
    if using_averaged_params:
        print(f"Camera params: PnP averaged over multiple pedestrians")
    else:
        print(f"Camera params: PnP per-pedestrian estimation")
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
            
    print("\nNote: Camera parameters are estimated using PnP from 2D-3D correspondences.")
    print("      Averaged params use all pedestrians for more robust camera estimation.")
    print("      This approach requires GT but validates triangulation quality.")
    
    # Return mean error for analysis
    if aligned_keypoints and gt_keypoints_3d:
        errors = []
        for name in aligned_keypoints.keys():
            if name in gt_keypoints_3d:
                error = np.linalg.norm(aligned_keypoints[name] - gt_keypoints_3d[name])
                errors.append(error)
        return np.mean(errors) if errors else float('inf')
    return float('inf')


def compute_frame_error(basedir_2d, basedir_3d, capture_date, frame_id, track_id, K, P1=None, P2=None):
    """
    Compute the triangulation error for a single pedestrian in a frame.
    
    Returns:
        mean_error: Mean error across all keypoints (in meters), or inf if computation failed
    """
    # Load 2D keypoints from stereo pair
    keypoints_blu = load_2d_keypoints(basedir_2d, capture_date, 'blu79CF', frame_id, track_id)
    keypoints_grn = load_2d_keypoints(basedir_2d, capture_date, 'grn43E3', frame_id, track_id)
    
    if keypoints_blu is None or keypoints_grn is None:
        return float('inf')
    
    # Load 3D ground truth mesh
    mesh_3d = load_3d_mesh(basedir_3d, capture_date, frame_id, track_id)
    if mesh_3d is None:
        return float('inf')
    
    # Extract ground truth keypoints from mesh
    gt_keypoints_3d = extract_keypoints_from_mesh(mesh_3d)
    
    # Estimate camera params if not provided
    if P1 is None or P2 is None:
        R1, t1, P1, success1 = estimate_camera_params_pnp(keypoints_blu, gt_keypoints_3d, K)
        R2, t2, P2, success2 = estimate_camera_params_pnp(keypoints_grn, gt_keypoints_3d, K)
        if not success1 or not success2:
            return float('inf')
    
    # Triangulate using camera parameters
    triangulated_keypoints = triangulate_skeleton(keypoints_blu, keypoints_grn, P1, P2)
    
    if not triangulated_keypoints or not gt_keypoints_3d:
        return float('inf')
    
    # Get common keypoints
    common_keys = [k for k in triangulated_keypoints.keys() if k in gt_keypoints_3d]
    
    if len(common_keys) < 4:
        return float('inf')
    
    tri_points = np.array([triangulated_keypoints[k] for k in common_keys])
    gt_points = np.array([gt_keypoints_3d[k] for k in common_keys])
    
    # Align using Procrustes
    aligned_points = procrustes_alignment(tri_points, gt_points)
    
    # Compute errors
    errors = np.linalg.norm(aligned_points - gt_points, axis=1)
    return np.mean(errors)


def find_frames_with_least_error(basedir_2d, basedir_3d, capture_date, n_best=10):
    """
    Analyze all frames and find those with the least triangulation error.
    
    Args:
        basedir_2d: Path to 2D labels directory
        basedir_3d: Path to 3D labels directory
        capture_date: Capture date string
        n_best: Number of best frames to return
    
    Returns:
        List of tuples (frame_id, track_id, mean_error) sorted by error
    """
    K, _, _ = get_camera_parameters()
    
    # Find all available frames
    pattern = os.path.join(basedir_2d, f'{capture_date}_blu79CF_*.json')
    files = glob.glob(pattern)
    
    # Extract unique frame IDs using the helper function
    frame_ids = set()
    for f in files:
        fid, _ = parse_label_filename(f)
        if fid is not None:
            frame_ids.add(fid)
    
    frame_ids = sorted(frame_ids)
    print(f"Found {len(frame_ids)} unique frames to analyze")
    
    # Collect errors for all frame-track combinations
    all_errors = []
    
    for i, frame_id in enumerate(frame_ids):
        if (i + 1) % 20 == 0:
            print(f"Processing frame {i+1}/{len(frame_ids)}...")
        
        # Find track IDs for this frame
        pattern = os.path.join(basedir_2d, f'{capture_date}_blu79CF_{frame_id:07d}_*.json')
        frame_files = glob.glob(pattern)
        track_ids = [parse_label_filename(f)[1] for f in frame_files if parse_label_filename(f)[1] is not None]
        
        for track_id in track_ids:
            # Check that both cameras have this track
            grn_file = os.path.join(basedir_2d, f'{capture_date}_grn43E3_{frame_id:07d}_{track_id}.json')
            if not os.path.exists(grn_file):
                continue
            
            # Check that 3D ground truth exists
            mesh_file = os.path.join(basedir_3d, f'{capture_date}_{frame_id:07d}_{track_id}.ply')
            if not os.path.exists(mesh_file):
                continue
            
            error = compute_frame_error(basedir_2d, basedir_3d, capture_date, frame_id, track_id, K)
            if error != float('inf'):
                all_errors.append((frame_id, track_id, error))
    
    # Sort by error
    all_errors.sort(key=lambda x: x[2])
    
    print(f"\nAnalyzed {len(all_errors)} frame-track combinations")
    
    return all_errors[:n_best]


def find_common_pedestrians(basedir_2d, basedir_3d, capture_date, frame_id):
    """
    Find all pedestrians that appear in both cameras (blu79CF and grn43E3)
    and have 3D ground truth available.
    
    Args:
        basedir_2d: Path to 2D labels directory
        basedir_3d: Path to 3D labels directory  
        capture_date: Capture date string
        frame_id: Frame number
    
    Returns:
        List of track_ids that appear in both cameras with 3D ground truth
    """
    # Find all track IDs in blu79CF camera
    pattern_blu = os.path.join(basedir_2d, f'{capture_date}_blu79CF_{frame_id:07d}_*.json')
    blu_files = glob.glob(pattern_blu)
    blu_track_ids = set()
    for f in blu_files:
        _, tid = parse_label_filename(f)
        if tid is not None:
            blu_track_ids.add(tid)
    
    # Find all track IDs in grn43E3 camera
    pattern_grn = os.path.join(basedir_2d, f'{capture_date}_grn43E3_{frame_id:07d}_*.json')
    grn_files = glob.glob(pattern_grn)
    grn_track_ids = set()
    for f in grn_files:
        _, tid = parse_label_filename(f)
        if tid is not None:
            grn_track_ids.add(tid)
    
    # Find intersection (pedestrians in both cameras)
    common_track_ids = blu_track_ids & grn_track_ids
    
    # Filter to only those with 3D ground truth
    valid_track_ids = []
    for tid in common_track_ids:
        mesh_file = os.path.join(basedir_3d, f'{capture_date}_{frame_id:07d}_{tid}.ply')
        if os.path.exists(mesh_file):
            valid_track_ids.append(tid)
    
    return valid_track_ids


def get_all_frame_ids(basedir_2d, capture_date):
    """
    Get all unique frame IDs from the dataset.
    
    Returns:
        Sorted list of frame IDs
    """
    pattern = os.path.join(basedir_2d, f'{capture_date}_blu79CF_*.json')
    files = glob.glob(pattern)
    
    frame_ids = set()
    for f in files:
        fid, _ = parse_label_filename(f)
        if fid is not None:
            frame_ids.add(fid)
    
    return sorted(frame_ids)


def triangulate_all_pedestrians_in_frame(basedir_2d, basedir_3d, capture_date, frame_id, P1, P2):
    """
    Triangulate all pedestrians visible in both cameras for a given frame.
    
    Args:
        basedir_2d: Path to 2D labels directory
        basedir_3d: Path to 3D labels directory
        capture_date: Capture date string
        frame_id: Frame number
        P1, P2: Projection matrices for cameras
    
    Returns:
        dict: {track_id: {'triangulated': keypoints_3d, 'gt': gt_keypoints_3d}}
    """
    # Find pedestrians visible in both cameras with 3D GT
    track_ids = find_common_pedestrians(basedir_2d, basedir_3d, capture_date, frame_id)
    
    results = {}
    for track_id in track_ids:
        # Load 2D keypoints from both cameras
        keypoints_blu = load_2d_keypoints(basedir_2d, capture_date, 'blu79CF', frame_id, track_id)
        keypoints_grn = load_2d_keypoints(basedir_2d, capture_date, 'grn43E3', frame_id, track_id)
        
        if keypoints_blu is None or keypoints_grn is None:
            continue
        
        # Load 3D ground truth mesh
        mesh_3d = load_3d_mesh(basedir_3d, capture_date, frame_id, track_id)
        if mesh_3d is None:
            continue
        
        # Extract ground truth keypoints
        gt_keypoints_3d = extract_keypoints_from_mesh(mesh_3d)
        
        # Triangulate from 2D observations
        triangulated_keypoints = triangulate_skeleton(keypoints_blu, keypoints_grn, P1, P2)
        
        if triangulated_keypoints and gt_keypoints_3d:
            # Apply Procrustes alignment for fair comparison
            common_keys = [k for k in triangulated_keypoints.keys() if k in gt_keypoints_3d]
            
            if len(common_keys) >= 4:
                tri_points = np.array([triangulated_keypoints[k] for k in common_keys])
                gt_points = np.array([gt_keypoints_3d[k] for k in common_keys])
                
                # Align using Procrustes
                aligned_points = procrustes_alignment(tri_points, gt_points)
                
                # Create aligned keypoints dict
                aligned_keypoints = {}
                for i, name in enumerate(common_keys):
                    aligned_keypoints[name] = aligned_points[i]
                
                results[track_id] = {
                    'triangulated': aligned_keypoints,
                    'gt': gt_keypoints_3d
                }
    
    return results


def visualize_frame_3d_scene(basedir_2d, basedir_3d, capture_date, frame_id, P1, P2, 
                              output_file=None, fig_size=(14, 10)):
    """
    Create a 3D scene visualization for a single frame showing all pedestrians.
    Ground truth shown in green, triangulated poses shown in blue.
    
    Args:
        basedir_2d: Path to 2D labels directory
        basedir_3d: Path to 3D labels directory
        capture_date: Capture date string
        frame_id: Frame number
        P1, P2: Projection matrices for cameras
        output_file: Optional path to save the figure
        fig_size: Figure size tuple
    
    Returns:
        numpy array of the rendered figure as an image
    """
    # Triangulate all pedestrians in this frame
    all_pedestrians = triangulate_all_pedestrians_in_frame(
        basedir_2d, basedir_3d, capture_date, frame_id, P1, P2
    )
    
    # Create the figure
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection='3d')
    
    # Collect all points to set axis limits
    all_points = []
    
    # Colors for different pedestrians (for visual distinction)
    ped_colors_gt = plt.cm.Greens(np.linspace(0.4, 0.9, max(len(all_pedestrians), 1)))
    ped_colors_tri = plt.cm.Blues(np.linspace(0.4, 0.9, max(len(all_pedestrians), 1)))
    
    # Plot each pedestrian
    for idx, (track_id, data) in enumerate(all_pedestrians.items()):
        gt_keypoints = data['gt']
        tri_keypoints = data['triangulated']
        
        # Ground truth skeleton (green shades)
        color_gt = ped_colors_gt[idx % len(ped_colors_gt)][:3]
        plot_skeleton_3d(ax, gt_keypoints, color=color_gt, label=f'GT', 
                        marker='o', linewidth=2)
        
        # Triangulated skeleton (blue shades)
        color_tri = ped_colors_tri[idx % len(ped_colors_tri)][:3]
        plot_skeleton_3d(ax, tri_keypoints, color=color_tri, label=f'Tri',
                        marker='^', linewidth=1.5)
        
        # Collect points for axis limits
        all_points.extend(list(gt_keypoints.values()))
        all_points.extend(list(tri_keypoints.values()))
    
    # Set axis limits based on all points
    if all_points:
        all_points_arr = np.array(all_points)
        center = np.mean(all_points_arr, axis=0)
        max_range = np.max(np.abs(all_points_arr - center)) * 1.3
        
        ax.set_xlim([center[0] - max_range, center[0] + max_range])
        ax.set_ylim([center[1] - max_range, center[1] + max_range])
        ax.set_zlim([center[2] - max_range, center[2] + max_range])
    
    # Labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'3D Pose: Frame {frame_id} | {len(all_pedestrians)} pedestrians\n'
                 f'Green=Ground Truth, Blue=Triangulated', fontsize=12)
    
    # Set viewing angle
    ax.view_init(elev=20, azim=45)
    
    # Add legend (simplified - just one entry for GT and Tri)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='green', marker='o', linestyle='-', label='Ground Truth'),
        Line2D([0], [0], color='blue', marker='^', linestyle='-', label='Triangulated')
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    
    plt.tight_layout()
    
    # Convert figure to image array
    # Note: Using buffer_rgba() instead of deprecated tostring_rgb() for matplotlib >= 3.8
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img_array = np.asarray(buf)
    # Convert RGBA to RGB
    img_array = img_array[:, :, :3]
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved frame to: {output_file}")
    
    plt.close(fig)
    
    return img_array


def generate_video(basedir_2d, basedir_3d, capture_date, output_video='triangulation_video.mp4',
                   fps=10, fig_size=(14, 10)):
    """
    Generate a video showing 3D triangulated poses vs ground truth for all frames.
    
    Args:
        basedir_2d: Path to 2D labels directory
        basedir_3d: Path to 3D labels directory
        capture_date: Capture date string
        output_video: Output video filename
        fps: Frames per second for the video
        fig_size: Figure size for each frame
    """
    print("="*60)
    print("Generating 3D Triangulation Video")
    print("="*60)
    
    # Get rectified projection matrices from calibration
    P1, P2, R_rect_00, R_rect_01, Q = get_stereo_rectified_parameters()
    
    print(f"Using calibration from ./data/calib/calib_cam_to_cam_blu79CF-grn43E3.txt")
    print(f"Projection matrix P1 (blu79CF):\n{P1}")
    print(f"Projection matrix P2 (grn43E3):\n{P2}")
    
    # Get all frame IDs
    frame_ids = get_all_frame_ids(basedir_2d, capture_date)
    print(f"\nFound {len(frame_ids)} frames to process")
    print(f"Frame range: {min(frame_ids)} to {max(frame_ids)}")
    
    # Create output directory for frames
    frames_dir = './output_frames'
    os.makedirs(frames_dir, exist_ok=True)
    
    # Collect frames with valid pedestrians
    valid_frames = []
    frame_images = []
    
    for i, frame_id in enumerate(frame_ids):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"Processing frame {i+1}/{len(frame_ids)} (frame_id={frame_id})...")
        
        # Check if frame has any valid pedestrians
        track_ids = find_common_pedestrians(basedir_2d, basedir_3d, capture_date, frame_id)
        
        if len(track_ids) > 0:
            # Generate the 3D scene image
            frame_file = os.path.join(frames_dir, f'frame_{frame_id:07d}.png')
            img_array = visualize_frame_3d_scene(
                basedir_2d, basedir_3d, capture_date, frame_id, P1, P2,
                output_file=frame_file, fig_size=fig_size
            )
            
            valid_frames.append(frame_id)
            frame_images.append(img_array)
    
    print(f"\nGenerated {len(valid_frames)} frames with valid pedestrians")
    
    if len(frame_images) == 0:
        print("ERROR: No valid frames found!")
        return
    
    # Get frame dimensions
    height, width = frame_images[0].shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    print(f"\nWriting video: {output_video}")
    print(f"Resolution: {width}x{height}, FPS: {fps}")
    
    for img in frame_images:
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        video_writer.write(img_bgr)
    
    video_writer.release()
    
    print(f"\nVideo saved to: {output_video}")
    print(f"Total frames: {len(frame_images)}")
    print(f"Duration: {len(frame_images)/fps:.1f} seconds")
    print("="*60)
    
    return output_video


def main():
    """Main entry point."""
    # Base directories for labels
    basedir_2d = './labels-2d_20171207T2024/20171207T2024'
    basedir_3d = './labels-3d-pose_20171207T2024/20171207T2024'
    
    # Demo parameters
    capture_date = '20171207T2024'
    
    print("="*60)
    print("PedX 3D Pose Triangulation with Calibration Data")
    print("="*60)
    
    # Generate the video showing all frames
    output_video = generate_video(
        basedir_2d, basedir_3d, capture_date,
        output_video='triangulation_video.mp4',
        fps=10,
        fig_size=(14, 10)
    )
    
    print("\n" + "="*60)
    print("Processing complete!")
    print(f"Output video: {output_video}")
    print("="*60)


if __name__ == '__main__':
    main()
