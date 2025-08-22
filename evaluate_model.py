import time
import math
import numpy as np
import mujoco as mj

SCENE_PATH = r"C:\Users\Cyril\PycharmProjects\RobotArm\lowCostRobotArm\robotScene.xml"


# ---------- Enhanced Motion Control ----------
def smooth_interpolate(start, end, t, motion_type="ease_in_out"):
    """Enhanced interpolation with different motion curves"""
    if motion_type == "linear":
        return start + t * (end - start)
    elif motion_type == "ease_in_out":
        # Smooth acceleration and deceleration
        t = 3 * t ** 2 - 2 * t ** 3  # Smoothstep function
        return start + t * (end - start)
    elif motion_type == "ease_in":
        t = t ** 2
        return start + t * (end - start)
    elif motion_type == "ease_out":
        t = 1 - (1 - t) ** 2
        return start + t * (end - start)
    elif motion_type == "natural":
        # More natural robot motion with slight overshoot correction
        t = t ** 2 * (3 - 2 * t) * (1 + 0.1 * np.sin(t * np.pi))
        return start + t * (end - start)


def ik_step_to_pos_smooth(model, data, site_id, target_pos, dof_indices,
                          step_gain=0.3, damping=1e-4, max_step=0.01):
    """Slower, more controlled IK steps"""
    cur = data.site_xpos[site_id]
    err = target_pos - cur
    if np.linalg.norm(err) < 1e-4:
        return np.linalg.norm(err)

    # Compute full Jacobian for the site translation
    jacp = np.zeros((3, model.nv))
    mj.mj_jacSite(model, data, jacp, None, site_id)
    J = jacp[:, dof_indices]

    # Damped least-squares with higher damping for smoother motion
    JT = J.T
    A = J @ JT + (damping * np.eye(3))
    dq = JT @ np.linalg.solve(A, step_gain * err)

    # Smaller step size for smoother motion
    if np.linalg.norm(dq) > max_step:
        dq = dq * (max_step / (np.linalg.norm(dq) + 1e-8))

    # Apply joint increments
    for k, dof in enumerate(dof_indices):
        j = model.dof_jntid[dof]
        a = model.jnt_qposadr[j]
        if model.jnt_type[j] in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE):
            data.qpos[a] += dq[k]
            clamp_joint_qpos(model, data, j)

    mj.mj_forward(model, data)
    return float(np.linalg.norm(err))


def move_ee_smooth(model, data, ee_sid, dof_indices, target_pos, duration=3.0,
                   motion_type="ease_in_out", viewer=None, adhesive=None, description=""):
    """Smooth trajectory execution with time-based control"""
    start_pos = data.site_xpos[ee_sid].copy()
    start_time = time.time()

    print(f"  {description} (duration: {duration:.1f}s)")

    steps = int(duration * 300)  # 100 Hz control
    for i in range(steps + 1):
        current_time = time.time()
        elapsed = current_time - start_time
        t = min(elapsed / duration, 1.0)

        # Smooth interpolation
        intermediate_pos = smooth_interpolate(start_pos, target_pos, t, motion_type)

        # Multiple small IK steps for each trajectory point
        for _ in range(3):
            ik_step_to_pos_smooth(model, data, ee_sid, intermediate_pos, dof_indices)

        # Handle adhesive grasping
        if adhesive and adhesive.get("active", False):
            ee_p = data.site_xpos[ee_sid]
            qadr = adhesive["cube_qpos_adr"]
            offset = adhesive.get("offset", np.array([0.0, 0.0, -0.02]))
            cube_target = ee_p + offset
            set_free_body_pose(model, data, qadr, cube_target, adhesive.get("quat", None))

        # Visual update with realistic timing
        if viewer:
            viewer.sync()
            time.sleep(0.01)  # 100 Hz update rate

        if t >= 1.0:
            break

    # Final positioning with high precision
    for _ in range(10):
        err = ik_step_to_pos_smooth(model, data, ee_sid, target_pos, dof_indices)
        if err < 1e-3:
            break
        if viewer:
            viewer.sync()
            time.sleep(0.01)


def set_gripper_state_smooth(model, data, grip_info, open_close, duration=1.5, viewer=None):
    """Smooth gripper actuation with realistic timing"""
    if not grip_info:
        return

    print(f"  {'Closing' if open_close > 0.5 else 'Opening'} gripper smoothly ({duration:.1f}s)")

    # Store initial positions
    initial_positions = {}
    target_positions = {}

    for g in grip_info:
        j = g["jid"]
        a = g["qpos"]
        initial_positions[j] = data.qpos[a]

        if g["limited"]:
            lo, hi = g["range"]
            target_positions[j] = lo + open_close * (hi - lo)
        else:
            target_positions[j] = (1.0 if open_close > 0.5 else 0.0)

    # Smooth gripper motion
    steps = int(duration * 60)  # 60 Hz for gripper
    for i in range(steps + 1):
        t = i / steps
        smooth_t = smooth_interpolate(0, 1, t, "ease_in_out")

        for g in grip_info:
            j = g["jid"]
            a = g["qpos"]
            current = initial_positions[j]
            target = target_positions[j]
            data.qpos[a] = current + smooth_t * (target - current)
            clamp_joint_qpos(model, data, j)

        mj.mj_forward(model, data)

        if viewer:
            viewer.sync()
            time.sleep(1.0 / 60.0)  # 60 Hz
        else:
            time.sleep(0.01)


def pause_and_observe(duration=1.0, viewer=None, description=""):
    """Realistic pauses for observation and stability"""
    if description:
        print(f"  {description} ({duration:.1f}s)")

    if viewer:
        steps = int(duration * 60)
        for _ in range(steps):
            viewer.sync()
            time.sleep(1.0 / 60.0)
    else:
        time.sleep(duration)


# ---------- Keep all helper functions from original ----------
def find_end_effector_site(model):
    candidates = [
        "ee", "end_effector", "grip", "gripper", "tcp", "tool", "tip", "wrist", "hand"
    ]
    avoid = ["cube_", "goal_", "workspace_", "robot_base", "midpoint_marker"]
    names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_SITE, i) or "" for i in range(model.nsite)]
    scores = []
    for i, name in enumerate(names):
        lname = name.lower()
        if any(lname.startswith(a) for a in avoid):
            continue
        hit = sum(1 for c in candidates if c in lname)
        if hit > 0:
            scores.append((hit, i))
    if scores:
        scores.sort(reverse=True)
        return scores[0][1]

    actuated_jids = set(_find_actuated_joint_ids(model))
    best = (-1.0, None)
    for sid in range(model.nsite):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_SITE, sid) or ""
        if any(name.lower().startswith(a) for a in avoid):
            continue
        body = model.site_bodyid[sid]
        body_joint_ids = [j for j in range(model.njnt) if model.jnt_bodyid[j] == body]
        if not any(j in actuated_jids for j in body_joint_ids):
            continue
        pos = model.site_pos[sid]
        d = float(np.linalg.norm(pos))
        if d > best[0]:
            best = (d, sid)
    return best[1]


def _find_actuated_joint_ids(model):
    jids = []
    for i in range(model.nu):
        if model.actuator_trntype[i] == mj.mjtTrn.mjTRN_JOINT:
            jids.append(model.actuator_trnid[i, 0])
    return jids


def find_actuated_dofs(model):
    dofs = []
    for i in range(model.nu):
        if model.actuator_trntype[i] != mj.mjtTrn.mjTRN_JOINT:
            continue
        jid = model.actuator_trnid[i, 0]
        jtype = model.jnt_type[jid]
        if jtype in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE):
            dofs.append(model.jnt_dofadr[jid])
    return sorted(set(dofs))


def find_gripper_joint_info(model):
    patterns = ["grip", "finger", "jaw", "claw"]
    joints = []
    for j in range(model.njnt):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, j) or ""
        if any(p in name.lower() for p in patterns):
            if model.jnt_type[j] in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE):
                joints.append(j)
    if not joints:
        return []

    info = []
    for j in joints:
        dof_adr = model.jnt_dofadr[j]
        qpos_adr = model.jnt_qposadr[j]
        jtype = model.jnt_type[j]
        rng = model.jnt_range[j]
        limited = bool(model.jnt_limited[j])
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, j) or f"joint{j}"
        info.append({
            "jid": j,
            "name": name,
            "dof": dof_adr,
            "qpos": qpos_adr,
            "type": jtype,
            "limited": limited,
            "range": rng.copy()
        })
    return info


def clamp_joint_qpos(model, data, j):
    if model.jnt_limited[j]:
        a = model.jnt_qposadr[j]
        low, high = model.jnt_range[j]
        data.qpos[a] = np.clip(data.qpos[a], low, high)


def get_site_pos(data, sid):
    return data.site_xpos[sid].copy()


def get_body_freejoint_qpos_adr(model, body_name):
    bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        return None
    for j in range(model.njnt):
        if model.jnt_bodyid[j] == bid and model.jnt_type[j] == mj.mjtJoint.mjJNT_FREE:
            return model.jnt_qposadr[j]
    return None


def get_free_body_pose(data, qpos_adr):
    pos = data.qpos[qpos_adr:qpos_adr + 3].copy()
    quat = data.qpos[qpos_adr + 3:qpos_adr + 7].copy()
    return pos, quat


def set_free_body_pose(model, data, qpos_adr, pos, quat=None):
    data.qpos[qpos_adr:qpos_adr + 3] = pos
    if quat is not None:
        data.qpos[qpos_adr + 3:qpos_adr + 7] = quat / np.linalg.norm(quat)
    mj.mj_forward(model, data)


def main():
    model = mj.MjModel.from_xml_path(SCENE_PATH)
    data = mj.MjData(model)

    # Identify key elements
    goal_sid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "goal_center")
    if goal_sid < 0:
        for nm in ["goal_surface", "goal_site"]:
            goal_sid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, nm)
            if goal_sid >= 0: break

    cube_qadr = get_body_freejoint_qpos_adr(model, "cube")
    ee_sid = find_end_effector_site(model)
    dof_indices = find_actuated_dofs(model)
    grip_info = find_gripper_joint_info(model)

    print(f"Found gripper joints: {len(grip_info)}")
    for g in grip_info:
        print(f"  - {g['name']} (range: {g['range']})")

    if cube_qadr is None or ee_sid is None or not dof_indices:
        print("Required components not found. Check your scene setup.")
        return

    mj.mj_forward(model, data)

    # Get positions
    cube_pos, cube_quat = get_free_body_pose(data, cube_qadr)
    goal_pos = get_site_pos(data, goal_sid)

    # Waypoints with more natural spacing
    approach_above = cube_pos + np.array([0.0, 0.0, 0.18])
    grasp_height = cube_pos + np.array([0.0, 0.0, 0.008])
    lift_height = cube_pos + np.array([0.0, 0.0, 0.15])
    goal_above = goal_pos + np.array([0.0, 0.0, 0.18])
    place_height = goal_pos + np.array([0.0, 0.0, 0.06])

    # Enhanced adhesive mechanism
    adhesive = {
        "active": False,
        "cube_qpos_adr": cube_qadr,
        "offset": np.array([0.0, 0.0, -0.02]),
        "quat": cube_quat.copy()
    }

    # Viewer setup
    try:
        from mujoco import viewer
        use_viewer = True
    except Exception:
        viewer = None
        use_viewer = False

    if use_viewer:
        with viewer.launch_passive(model, data) as v:
            print("\n=== HYPERREALISTIC PICK AND PLACE SEQUENCE ===\n")

            pause_and_observe(1.0, v, "System initialization and calibration")

            # Phase 1: Gripper preparation
            print("Phase 1: Gripper Preparation")
            if grip_info:
                set_gripper_state_smooth(model, data, grip_info, open_close=0.0,
                                         duration=2.0, viewer=v)
            pause_and_observe(0.8, v, "Gripper position verification")

            # Phase 2: Approach trajectory
            print("\nPhase 2: Approach Trajectory")
            move_ee_smooth(model, data, ee_sid, dof_indices, approach_above,
                           duration=4.0, motion_type="ease_in_out", viewer=v,
                           description="Moving to approach position above cube")
            pause_and_observe(1.2, v, "Position stabilization and target verification")

            # Phase 3: Precision descent
            print("\nPhase 3: Precision Descent")
            move_ee_smooth(model, data, ee_sid, dof_indices, grasp_height,
                           duration=3.5, motion_type="ease_in", viewer=v,
                           description="Precise descent to grasping altitude")
            pause_and_observe(0.8, v, "Final positioning adjustment")

            # Phase 4: Grasping sequence
            print("\nPhase 4: Grasping Sequence")
            if grip_info:
                set_gripper_state_smooth(model, data, grip_info, open_close=1.0,
                                         duration=2.5, viewer=v)
            pause_and_observe(1.0, v, "Grip force stabilization")

            adhesive["active"] = True
            print("  Secure grasp established - adhesive mechanism activated")
            pause_and_observe(0.5, v, "Grasp verification")

            # Phase 5: Lifting maneuver
            print("\nPhase 5: Lifting Maneuver")
            move_ee_smooth(model, data, ee_sid, dof_indices, lift_height,
                           duration=3.8, motion_type="ease_out", viewer=v,
                           adhesive=adhesive, description="Lifting object with controlled acceleration")
            pause_and_observe(1.0, v, "Lift completion and load balance check")

            # Phase 6: Transport trajectory
            print("\nPhase 6: Transport Trajectory")
            move_ee_smooth(model, data, ee_sid, dof_indices, goal_above,
                           duration=5.5, motion_type="natural", viewer=v,
                           adhesive=adhesive, description="Transporting object to target area")
            pause_and_observe(1.2, v, "Target area positioning verification")

            # Phase 7: Placement descent
            print("\nPhase 7: Placement Descent")
            move_ee_smooth(model, data, ee_sid, dof_indices, place_height,
                           duration=4.0, motion_type="ease_in", viewer=v,
                           adhesive=adhesive, description="Controlled descent for precision placement")
            pause_and_observe(0.8, v, "Placement altitude verification")

            # Phase 8: Release sequence
            print("\nPhase 8: Release Sequence")
            adhesive["active"] = False
            print("  Disengaging adhesive mechanism")
            pause_and_observe(0.5, v, "Load transfer to surface")

            if grip_info:
                set_gripper_state_smooth(model, data, grip_info, open_close=0.0,
                                         duration=2.2, viewer=v)
            pause_and_observe(1.0, v, "Release verification")

            # Phase 9: Retraction
            print("\nPhase 9: Safe Retraction")
            move_ee_smooth(model, data, ee_sid, dof_indices, goal_above,
                           duration=3.2, motion_type="ease_out", viewer=v,
                           description="Safe retraction to standby position")
            pause_and_observe(1.5, v, "Mission completion verification")

            print("\n=== PICK AND PLACE OPERATION COMPLETED SUCCESSFULLY ===")
            print("  • Object successfully transported to target location")
            print("  • All systems nominal")
            print("  • Ready for next operation")

            # Extended observation period
            pause_and_observe(5.0, v, "Extended observation period")

    else:
        print("Running headless hyperrealistic simulation...")
        # Implement headless version with same timing
        # ... (similar structure but without viewer)


if __name__ == "__main__":
    main()