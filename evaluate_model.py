import time
import math
import numpy as np
import mujoco as mj

SCENE_PATH = r"C:\Users\Cyril\PycharmProjects\RobotArm\lowCostRobotArm\robotScene.xml"


# ---------- Helpers ----------
def find_end_effector_site(model):
    # Prefer common EE site names
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
        # simple heuristic scoring
        hit = sum(1 for c in candidates if c in lname)
        if hit > 0:
            scores.append((hit, i))
    if scores:
        scores.sort(reverse=True)
        return scores[0][1]

    # Fallback: pick the site that belongs to a body with actuated joints and is far from world origin
    actuated_jids = set(_find_actuated_joint_ids(model))
    best = (-1.0, None)
    for sid in range(model.nsite):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_SITE, sid) or ""
        if any(name.lower().startswith(a) for a in avoid):
            continue
        body = model.site_bodyid[sid]
        # body has any joint that is actuated?
        body_joint_ids = [j for j in range(model.njnt) if model.jnt_bodyid[j] == body]
        if not any(j in actuated_jids for j in body_joint_ids):
            continue
        # approximate distance using default pos; refined later with data
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
    """
    Return the list of DoF indices (in nv space) that are actuated and belong
    to hinge/slide joints only. This avoids dealing with ball/free joints which
    need special handling for qpos (quaternions).
    """
    dofs = []
    for i in range(model.nu):
        if model.actuator_trntype[i] != mj.mjtTrn.mjTRN_JOINT:
            continue
        jid = model.actuator_trnid[i, 0]
        jtype = model.jnt_type[jid]
        if jtype in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE):
            dofs.append(model.jnt_dofadr[jid])  # 1 DoF per hinge/slide joint
    # de-duplicate and keep order
    return sorted(set(dofs))


def find_gripper_joint_info(model):
    # Look for joints that appear to belong to a gripper/fingers
    patterns = ["grip", "finger", "jaw", "claw"]
    joints = []
    for j in range(model.njnt):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, j) or ""
        if any(p in name.lower() for p in patterns):
            # Only hinge or slide are meaningful for open/close
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


def set_gripper_state(model, data, grip_info, open_close, viewer=None):
    # open_close in [0.0, 1.0], 0=open, 1=close
    if not grip_info:
        return
    for g in grip_info:
        j = g["jid"]
        a = g["qpos"]
        if g["limited"]:
            lo, hi = g["range"]
            target = lo + open_close * (hi - lo)
        else:
            # default heuristic range
            target = (1.0 if open_close > 0.5 else 0.0)
        # Gradually move to target
        current = data.qpos[a]
        for step in range(30):
            alpha = step / 29.0
            data.qpos[a] = current + alpha * (target - current)
            clamp_joint_qpos(model, data, j)
            mj.mj_forward(model, data)
            if viewer:
                viewer.sync()
                time.sleep(model.opt.timestep * 0.5)


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


def ik_step_to_pos(model, data, site_id, target_pos, dof_indices, step_gain=0.6, damping=1e-4, max_step=0.02):
    cur = data.site_xpos[site_id]
    err = target_pos - cur
    if np.linalg.norm(err) < 1e-4:
        return np.linalg.norm(err)
    # Compute full Jacobian for the site translation
    jacp = np.zeros((3, model.nv))
    mj.mj_jacSite(model, data, jacp, None, site_id)
    # Select columns for actuated dofs
    J = jacp[:, dof_indices]  # 3 x m
    # Damped least-squares
    JT = J.T
    A = J @ JT + (damping * np.eye(3))
    dq = JT @ np.linalg.solve(A, step_gain * err)
    # Limit step size
    if np.linalg.norm(dq) > max_step:
        dq = dq * (max_step / (np.linalg.norm(dq) + 1e-8))
    # Map dof increments to qpos increments (assumes 1-1 for hinge/slide)
    for k, dof in enumerate(dof_indices):
        j = model.dof_jntid[dof]
        a = model.jnt_qposadr[j]
        # Only hinge/slide joints have 1 dof mapping
        if model.jnt_type[j] in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE):
            data.qpos[a] += dq[k]
            clamp_joint_qpos(model, data, j)
    mj.mj_forward(model, data)
    return float(np.linalg.norm(err))


def move_ee_to(model, data, ee_sid, dof_indices, target_pos, iters=200, tol=2e-3, viewer=None, adhesive=None):
    for _ in range(iters):
        err = ik_step_to_pos(model, data, ee_sid, target_pos, dof_indices)

        # If "adhesive" grasp is active, keep cube attached to EE
        if adhesive and adhesive.get("active", False):
            ee_p = data.site_xpos[ee_sid]
            qadr = adhesive["cube_qpos_adr"]
            offset = adhesive.get("offset", np.array([0.0, 0.0, -0.02]))
            cube_target = ee_p + offset
            set_free_body_pose(model, data, qadr, cube_target, adhesive.get("quat", None))

        # visual step
        if viewer:
            viewer.sync()
            time.sleep(model.opt.timestep)
        if err < tol:
            break


def create_weld_constraint(model, data, body1_name, body2_name):
    """Create a weld constraint between two bodies"""
    body1_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body1_name)
    body2_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body2_name)

    if body1_id < 0 or body2_id < 0:
        return False

    # This is a simplified approach - in practice, you might need to modify
    # the XML or use contact constraints
    return True


def main():
    model = mj.MjModel.from_xml_path(SCENE_PATH)
    data = mj.MjData(model)

    # Identify key elements
    goal_sid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "goal_center")
    if goal_sid < 0:
        # fallback to goal_surface or goal_site
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

    if cube_qadr is None:
        print("Cube free joint not found. Make sure 'cube' has a freejoint.")
        return
    if ee_sid is None:
        print("End-effector site not found. Please add an EE site (e.g., name contains 'ee' or 'grip').")
        print("Available sites:", [mj.mj_id2name(model, mj.mjtObj.mjOBJ_SITE, i) for i in range(model.nsite)])
        return
    if not dof_indices:
        print("No actuated joints found. Cannot move the arm.")
        return

    # Initial forward
    mj.mj_forward(model, data)

    # Get current positions
    cube_pos, cube_quat = get_free_body_pose(data, cube_qadr)
    goal_pos = get_site_pos(data, goal_sid)

    # Define waypoints with better positioning
    approach_above = cube_pos + np.array([0.0, 0.0, 0.15])
    grasp_height = cube_pos + np.array([0.0, 0.0, 0.01])  # Lower for better grasping
    lift_height = cube_pos + np.array([0.0, 0.0, 0.12])
    goal_above = goal_pos + np.array([0.0, 0.0, 0.15])
    place_height = goal_pos + np.array([0.0, 0.0, 0.05])  # Higher for safe placement

    # Enhanced adhesive mechanism
    adhesive = {
        "active": False,
        "cube_qpos_adr": cube_qadr,
        "offset": np.array([0.0, 0.0, -0.02]),
        "quat": cube_quat.copy()
    }

    # Always use adhesive for reliable grasping
    use_adhesive = True

    # Viewer
    try:
        from mujoco import viewer
        use_viewer = True
    except Exception:
        viewer = None
        use_viewer = False

    if use_viewer:
        with viewer.launch_passive(model, data) as v:
            print("Starting pick and place sequence...")

            # Step 1: Open gripper first
            if grip_info:
                print("Opening gripper...")
                set_gripper_state(model, data, grip_info, open_close=0.0, viewer=v)
                time.sleep(0.5)

            # Step 2: Move to approach position
            print("Moving to approach position...")
            move_ee_to(model, data, ee_sid, dof_indices, approach_above, viewer=v)

            # Step 3: Descend to grasp height
            print("Descending to cube...")
            move_ee_to(model, data, ee_sid, dof_indices, grasp_height, viewer=v)

            # Step 4: Activate grasping
            print("Grasping cube...")
            if grip_info:
                set_gripper_state(model, data, grip_info, open_close=1.0, viewer=v)
                time.sleep(0.5)

            # Always activate adhesive for reliable transport
            adhesive["active"] = True
            print("Adhesive grasp activated")

            # Step 5: Lift cube
            print("Lifting cube...")
            move_ee_to(model, data, ee_sid, dof_indices, lift_height, viewer=v, adhesive=adhesive)
            time.sleep(0.5)

            # Step 6: Move over goal
            print("Moving to goal area...")
            move_ee_to(model, data, ee_sid, dof_indices, goal_above, viewer=v, adhesive=adhesive)

            # Step 7: Lower to place
            print("Lowering to place...")
            move_ee_to(model, data, ee_sid, dof_indices, place_height, viewer=v, adhesive=adhesive)

            # Step 8: Release
            print("Releasing cube...")
            adhesive["active"] = False
            if grip_info:
                set_gripper_state(model, data, grip_info, open_close=0.0, viewer=v)
                time.sleep(0.5)

            # Step 9: Retract
            print("Retracting arm...")
            move_ee_to(model, data, ee_sid, dof_indices, goal_above, viewer=v)

            print("Pick and place completed!")

            # Hold simulation for observation
            for _ in range(1000):
                v.sync()
                time.sleep(model.opt.timestep)
    else:
        # Headless execution with detailed logging
        print("Running headless simulation...")

        steps = [
            ("Open gripper", lambda: set_gripper_state(model, data, grip_info, 0.0) if grip_info else None),
            ("Approach", approach_above),
            ("Descend", grasp_height),
            ("Grasp", lambda: (set_gripper_state(model, data, grip_info, 1.0) if grip_info else None,
                               setattr(adhesive, 'active', True))[1]),
            ("Lift", lift_height),
            ("Move to goal", goal_above),
            ("Lower", place_height),
            ("Release", lambda: (setattr(adhesive, 'active', False),
                                 set_gripper_state(model, data, grip_info, 0.0) if grip_info else None)[0]),
            ("Retract", goal_above)
        ]

        for name, action in steps:
            print(f"Step: {name}")
            if callable(action):
                action()
                mj.mj_forward(model, data)
                time.sleep(0.02)
            else:
                move_ee_to(model, data, ee_sid, dof_indices, action,
                           adhesive=adhesive if adhesive.get("active") else None)


if __name__ == "__main__":
    main()