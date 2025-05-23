#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import submission  


def project_cad():
    # ----------------------------
    # Step 1: Load data from data/pnp.npz
    # ----------------------------
    # Load the npz file with allow_pickle=True, because CAD model is saved as an object.
    data = np.load('../data/pnp.npz', allow_pickle=True)
    image = data['image']  # Image (H x W x 3)
    
    # Load the CAD model
    # It might be stored as a dictionary or as a tuple (vertices, faces)
    cad_obj = data['cad'].item()  # This converts the stored object to a Python object
    if isinstance(cad_obj, dict):
        cad_vertices = np.array(cad_obj['vertices'], dtype=np.float32)
        cad_faces = cad_obj.get('faces', None)
    elif isinstance(cad_obj, (tuple, list)):
        cad_vertices = np.array(cad_obj[0], dtype=np.float32)
        cad_faces = cad_obj[1] if len(cad_obj) > 1 else None
    else:
        cad_vertices = np.array(cad_obj, dtype=np.float32)
        cad_faces = None

    # ----------------------------
    # Step 2: Load correspondences and estimate camera pose.
    # ----------------------------
    x = data['x']  # Given 2D points (M x 2)
    X = data['X']  # Corresponding 3D points (M x 3)
    
    # Estimate the camera matrix P using your DLT-based pose estimation
    P = submission.estimate_pose(x, X)
    # Decompose P to obtain intrinsic K, rotation R, and translation t.
    K, R, t = submission.estimate_params(P)

    # ----------------------------
    # Step 3: Use your estimated camera matrix P to project the given 3D points X onto the image.
    # ----------------------------
    # Convert 3D points to homogeneous coordinates (M x 4)
    M = X.shape[0]
    X_hom = np.hstack((X, np.ones((M, 1), dtype=X.dtype)))
    x_projected = (P @ X_hom.T).T  # (M x 3)
    x_projected = x_projected[:, :2] / x_projected[:, 2, np.newaxis]  # Divide by third coordinate to get 2D.

    # ----------------------------
    # Step 4: Plot the given 2D points x and the projected 3D points on screen.
    # ----------------------------
    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(image)
    ax1.set_title('Given 2D pts (green) vs. Reproj. 3D pts (black)')
    ax1.scatter(x[:, 0], x[:, 1], c='g', marker='o', label='Given 2D pts')
    ax1.scatter(x_projected[:, 0], x_projected[:, 1], c='k', marker='s', label='Reproj. 3D pts')
    ax1.legend()
    
    # Also, print mean reprojection error for diagnostic purposes
    reproj_error = np.linalg.norm(x - x_projected, axis=1).mean()
    print("Mean reprojection error: {:.4f} pixels".format(reproj_error))
    
    # ----------------------------
    # Step 5: Draw the CAD model rotated by your estimated rotation R on screen.
    # ----------------------------
    # Rotate the CAD vertices using R only (ignoring translation for this view)
    cad_rotated = (R @ cad_vertices.T).T  # (Nc x 3)
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    # If face information is available, draw a wireframe; otherwise, scatter the vertices.
    if cad_faces is not None:
        # Check if face indices need adjustment:
        max_index = np.max(np.concatenate([np.array(face) for face in cad_faces]))
        if max_index >= cad_rotated.shape[0]:
            # If max index equals the number of vertices, assume indices are 1-indexed
            adjusted_faces = [tuple(np.array(face) - 1) for face in cad_faces]
        else:
            adjusted_faces = cad_faces
        # Draw each face as a polygon edge loop.
        for face in adjusted_faces:
            face_indices = list(face) + [face[0]]
            x_vals = [cad_rotated[idx, 0] for idx in face_indices]
            y_vals = [cad_rotated[idx, 1] for idx in face_indices]
            z_vals = [cad_rotated[idx, 2] for idx in face_indices]
            ax2.plot(x_vals, y_vals, z_vals, color='b', linewidth=1.5)
    else:
        ax2.scatter(cad_rotated[:, 0], cad_rotated[:, 1], cad_rotated[:, 2], c='b', marker='o')
    ax2.set_title('Rotated CAD Model (blue)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.view_init(elev=20, azim=-60)  # Adjust the viewing angle if desired

    # ----------------------------
    # Step 6: Project all the CAD's vertices onto the image and draw the projected CAD model overlapping with the 2D image.
    # ----------------------------
    Nc = cad_vertices.shape[0]
    cad_hom = np.hstack((cad_vertices, np.ones((Nc, 1), dtype=cad_vertices.dtype)))  # (Nc, 4)
    cad_proj = (P @ cad_hom.T).T  # (Nc, 3)
    cad_proj = cad_proj[:, :2] / cad_proj[:, 2, np.newaxis]  # (Nc, 2)
    
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(image)
    ax3.set_title('Projected CAD Model (red) on image')
    # Draw the projected CAD as a wireframe if face info is available, else scatter the vertices.
    if cad_faces is not None:
        # Use the same adjusted faces as above
        def draw_wireframe_2d(ax, pts2d, faces, color='r', lw=1.5):
            for face in faces:
                face_indices = list(face) + [face[0]]
                x_vals = [pts2d[idx, 0] for idx in face_indices]
                y_vals = [pts2d[idx, 1] for idx in face_indices]
                ax.plot(x_vals, y_vals, color=color, linewidth=lw)
        draw_wireframe_2d(ax3, cad_proj, adjusted_faces, color='r', lw=1.5)
    else:
        ax3.scatter(cad_proj[:,0], cad_proj[:,1], c='r', marker='x')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    project_cad()
