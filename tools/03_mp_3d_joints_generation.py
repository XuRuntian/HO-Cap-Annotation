import os

os.environ["MPLBACKEND"] = "Agg"  # Disable matplotlib GUI backend

from hocap_annotation.utils import *
from hocap_annotation.loaders import SequenceLoader

# Set the start method to 'spawn'
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass


from hocap_annotation.triangulation import runner_hand_3d_joints_ransac_torch


def draw_handmarks_frame(rgb_images, handmarks, serials, save_path=None):
    vis_images = [
        draw_debug_image(
            rgb_image,
            hand_marks=handmarks[:, idx],
            draw_boxes=True,
            draw_hand_sides=True,
        )
        for idx, rgb_image in enumerate(rgb_images)
    ]
    vis_frame = draw_image_grid(
        images=vis_images, names=serials, facecolor="black", titlecolor="white"
    )
    if save_path is not None:
        write_rgb_image(save_path, vis_frame)
    return vis_frame


def complete_3d_joints_by_cubic(joints_3d, ratio=0.5):
    def calculate_bone_lengths(joints_3d):
        """
        Calculate the bone lengths from the parent-child joint relationships.

        Parameters:
        - joints_3d: Observed 3D joints, shape (N, 21, 3).

        Returns:
        - Bone lengths for each joint, shape (21,).
        """
        bone_lengths = np.zeros(21)
        for i in range(1, 21):  # Skip the root joint, which has no parent
            parent_idx = HAND_JOINT_PARENTS[i]
            if parent_idx >= 0:  # Valid parent index
                bone_lengths[i] = np.linalg.norm(
                    joints_3d[:, parent_idx] - joints_3d[:, i], axis=1
                ).mean()
        return bone_lengths

    def adjust_joints_to_bone_lengths(joints_3d, bone_lengths):
        """
        Adjust the positions of the joints to respect the given bone lengths.

        Parameters:
        - joints_3d: Interpolated 3D joints, shape (N, 21, 3).
        - bone_lengths: Bone lengths to enforce, shape (21,).
        """
        for i in range(1, 21):  # Skip the root joint
            parent_idx = HAND_JOINT_PARENTS[i]
            if parent_idx >= 0:  # Valid parent index
                direction = joints_3d[:, i] - joints_3d[:, parent_idx]
                # Normalize the direction vector
                direction /= np.linalg.norm(direction, axis=1, keepdims=True)
                joints_3d[:, i] = joints_3d[:, parent_idx] + direction * bone_lengths[i]

    hands, N, joints, coords = joints_3d.shape
    complete_joints = joints_3d.copy()
    for hand in range(hands):
        valid_frames = np.where(np.all(complete_joints[hand] != -1, axis=(1, 2)))[0]

        if len(valid_frames) < int(N * ratio):
            print(
                f"** Not enough valid frames for interpolation for hand-{hand}. (#frames: {len(valid_frames)}/{N})"
            )
            continue

        # Calculate average bone lengths from the first frame (assuming it's fully observed)
        bone_lengths = calculate_bone_lengths(joints_3d[hand, valid_frames])

        for joint in range(joints):
            for coord in range(coords):
                valid_coords = complete_joints[hand, valid_frames, joint, coord]
                cs = CubicSpline(valid_frames, valid_coords, bc_type="clamped")
                interpolated_coords = cs(np.arange(N))
                complete_joints[hand, :, joint, coord] = interpolated_coords
        # Adjust the completed joints to respect bone lengths
        adjust_joints_to_bone_lengths(complete_joints[hand, :], bone_lengths)

    return complete_joints


class HandJointsEstimator:
    def __init__(self, sequence_folder, debug=False) -> None:
        self._data_folder = Path(sequence_folder).resolve()
        self._save_folder = self._data_folder / "processed/hand_detection"

        self._logger = get_logger(
            __class__.__name__,
            "DEBUG" if debug else "INFO",
            self._save_folder / "mp_3d_joints_estimation.log",
        )

        # Load variables from sequence loader
        self._loader = SequenceLoader(sequence_folder, device=CFG.device)
        self._rs_serials = self._loader.rs_serials
        self._num_frames = self._loader.num_frames
        self._rs_width = self._loader.rs_width
        self._rs_height = self._loader.rs_height
        self._mano_sides = self._loader.mano_sides
        self._M = self._loader.M2world.cpu().numpy()
        self._num_cameras = len(self._rs_serials)

    def run_hand_joints_3d_estimation(self):
        self._logger.info(">>>>>>>>>> Running Hand 3D Joints Estimation <<<<<<<<<<")
        start_time = time.time()

        self._save_folder.mkdir(parents=True, exist_ok=True)
        handmarks_file = self._save_folder / "mp_handmarks_results.npz"
        if not handmarks_file.exists():
            self._logger.error("*** Hand 2D Handmarks does not exist!!! ***")
            return

        joints_3d_file = self._save_folder / "mp_joints_3d_raw.npy"
        if joints_3d_file.exists():
            self._logger.info("Hand 3D Joints Results found, loading...")
            hand_joints_3d = np.load(joints_3d_file)
        else:
            self._logger.info("Start Hand 3D Joints Estimation...")
            # Load mp_handmarks
            handmarks = self._load_handmarks(handmarks_file)

            hand_joints_3d = [None] * self._num_frames
            tqbar = tqdm(total=self._num_frames, ncols=100)
            with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
                futures = {
                    executor.submit(
                        runner_hand_3d_joints_ransac_torch,
                        handmarks[:, frame_id],
                        self._M,
                        threshold=14.0,
                        iterations=500,
                        lr=1e-3,
                    ): frame_id
                    for frame_id in range(self._num_frames)
                }
                for future in concurrent.futures.as_completed(futures):
                    frame_id = futures[future]
                    try:
                        hand_joints_3d[frame_id] = future.result()
                    except Exception as e:
                        self._logger.error(f"Error in processing frame {frame_id}: {e}")
                    finally:
                        tqbar.update()
                futures.clear()
            tqbar.close()

            hand_joints_3d = np.stack(
                hand_joints_3d, axis=1, dtype=np.float32
            )  # (2, num_frames, num_joints, 3)
            if "right" not in self._mano_sides:
                hand_joints_3d[0] = -1
            if "left" not in self._mano_sides:
                hand_joints_3d[1] = -1
            np.save(joints_3d_file, hand_joints_3d)
            self._logger.debug(f"hand_joints_3d: {hand_joints_3d.shape}")

        hand_joints_3d = complete_3d_joints_by_cubic(hand_joints_3d)
        # hand_joints_3d = complete_3d_joints_by_linear(hand_joints_3d)
        np.save(self._save_folder / "mp_joints_3d_interpolated.npy", hand_joints_3d)

        # Project 3D hand joints to 2D
        self._logger.info("Projecting 3D Hand Joints to 2D...")
        hand_joints_2d = self._project_joints_3d_to_2d(hand_joints_3d)
        np.savez_compressed(
            self._save_folder / "mp_joints_3d_projection.npz", **hand_joints_2d
        )

        # Get 2D hand joints bbox
        self._logger.info("Calculating 2D hand joints bbox...")
        hand_joints_bbox = {
            serial: np.stack(
                [
                    [
                        get_bbox_from_landmarks(
                            hand_joints_2d[serial][hand_ind, frame_id],
                            self._rs_width,
                            self._rs_height,
                            10,
                        )
                        for frame_id in range(self._num_frames)
                    ]
                    for hand_ind in range(2)
                ]
            ).astype(np.int64)
            for serial in self._rs_serials
        }
        np.savez_compressed(
            self._save_folder / "mp_joints_3d_bbox.npz", **hand_joints_bbox
        )

    def render_hand_joints_3d(self):
        self._logger.info(">>>>>>>>>> Drawing Hand Joints 3D Results <<<<<<<<<<")
        t_start = time.time()

        hand_joints_2d_file = self._save_folder / "mp_joints_3d_projection.npz"
        if not hand_joints_2d_file.exists():
            self._logger.error("*** Hand 3D Joints Projection does not exist!!! ***")
            return
        hand_joints_2d = self._load_handmarks(hand_joints_2d_file)

        self._logger.info("Saving vis images...")
        save_vis_folder = self._save_folder / "vis" / "mp_joints_3d"
        save_vis_folder.mkdir(parents=True, exist_ok=True)

        tqbar = tqdm(total=self._num_frames, ncols=100)
        with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(
                    draw_handmarks_frame,
                    self._get_rgb_images_by_frame_id(frame_id),
                    hand_joints_2d[:, frame_id],
                    self._rs_serials,
                    save_vis_folder / f"vis_{frame_id:06d}.png",
                ): frame_id
                for frame_id in range(self._num_frames)
            }
            for future in concurrent.futures.as_completed(futures):
                frame_id = futures[future]
                try:
                    future.result()
                except Exception as e:
                    self._logger.error(f"Error in processing frame {frame_id}: {e}")
                finally:
                    tqbar.update()
            futures.clear()
        tqbar.close()

        self._logger.info("Saving vis video...")
        save_video_path = self._save_folder / "vis_mp_hand_joints_3d.mp4"
        image_files = [
            save_vis_folder / f"vis_{frame_id:06d}.png"
            for frame_id in range(self._num_frames)
        ]
        create_video_from_image_files(save_video_path, image_files, fps=30)

        self._logger.info(
            f">>>>>>>>>> Done!!! ({time.time() - t_start:.2f} seconds)<<<<<<<<<<"
        )

    def _get_rgb_images_by_frame_id(self, frame_id):
        rgb_images = [None] * self._num_cameras
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self._loader.get_rgb_image, frame_id, serial): idx
                for idx, serial in enumerate(self._rs_serials)
            }
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                rgb_images[idx] = future.result()
            futures.clear()
        return rgb_images

    def _load_handmarks(self, handmarks_file):
        self._logger.debug(f"Loading 2D handmarks from {handmarks_file}")
        handmarks = np.load(handmarks_file)
        handmarks = np.stack(
            [handmarks[s] for s in self._rs_serials], axis=2
        )  # (num_hands, num_frames, num_cameras, num_joints, 2)
        if "right" not in self._mano_sides:
            handmarks[0] = -1
        if "left" not in self._mano_sides:
            handmarks[1] = -1
        self._logger.debug(f"Handmarks loaded: {handmarks.shape}, {handmarks.dtype}")
        return handmarks

    def _project_joints_3d_to_2d(self, hand_joints_3d):
        self._logger.info("Projecting Hand 3D Joints to 2D...")
        hand_joints_2d = np.stack(
            [
                [
                    self._points_3d_to_2d(hand_joints_3d[hand_ind, frame_id], self._M)
                    for frame_id in range(self._num_frames)
                ]
                for hand_ind in range(2)
            ],
            axis=0,
            dtype=np.int64,
        )  # (2, num_frames, num_cameras, num_joints, 2)
        self._logger.debug(f"hand_joints_2d: {hand_joints_2d.shape}")

        hand_joints_2d = {
            serial: hand_joints_2d[:, :, cam_idx]
            for cam_idx, serial in enumerate(self._rs_serials)
        }
        return hand_joints_2d

    def _point_3d_to_2d(self, point_3d, M):
        point_2d = M @ np.append(point_3d, 1)
        point_2d /= point_2d[2]
        point_2d = point_2d[:2]
        # Check if the point is within the image
        if (
            point_2d[0] < 0
            or point_2d[0] >= self._rs_width
            or point_2d[1] < 0
            or point_2d[1] >= self._rs_height
        ):
            return np.array([-1, -1])
        return point_2d

    def _points_3d_to_2d(self, points_3d, Ms):
        points_2d = np.full((len(Ms), len(points_3d), 2), -1)
        for i, p_3d in enumerate(points_3d):
            if np.any(p_3d == -1):
                continue
            points_2d[:, i] = [self._point_3d_to_2d(p_3d, M) for M in Ms]
        return points_2d.astype(np.int64)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MP Hand 3D Joints Estimation")
    parser.add_argument(
        "--sequence_folder", type=str, default=None, help="Path to the sequence folder."
    )
    parser.add_argument("--debug", action="store_true", help="debug mode")
    args = parser.parse_args()

    if args.sequence_folder is None:
        raise ValueError("Please provide the sequence folder path.")

    estimator = HandJointsEstimator(args.sequence_folder, args.debug)
    estimator.run_hand_joints_3d_estimation()
    estimator.render_hand_joints_3d()
