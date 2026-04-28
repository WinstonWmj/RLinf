# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PrivilegedObsTerm:
    name: str
    dim: int
    scale: float = 1.0


PRIVILEGED_TEACHER_OBS_SPEC: tuple[PrivilegedObsTerm, ...] = (
    PrivilegedObsTerm("base_lin_vel", 3, 1.0),
    PrivilegedObsTerm("base_ang_vel", 3, 1.0),
    PrivilegedObsTerm("projected_gravity", 3, 1.0),
    PrivilegedObsTerm("actions", 31, 1.0),
    PrivilegedObsTerm("stage", 5, 1.0),
    PrivilegedObsTerm("delta_actions", 11, 1.0),
    PrivilegedObsTerm("dof_pos", 43, 1.0),
    PrivilegedObsTerm("dof_vel", 43, 0.05),
    PrivilegedObsTerm("placement_pos", 2, 1.0),
    PrivilegedObsTerm("table_pelvis_transform", 9, 1.0),
    PrivilegedObsTerm("hold_fingers_tips_force", 12, 1.0),
    PrivilegedObsTerm("hold_obj_transform", 9, 1.0),
    PrivilegedObsTerm("hold_hand_object_transform", 9, 1.0),
    PrivilegedObsTerm("target_place_pos", 3, 1.0),
    PrivilegedObsTerm("grasp_fingers_tips_force", 12, 1.0),
    PrivilegedObsTerm("grasp_obj_transform", 9, 1.0),
    PrivilegedObsTerm("grasp_hand_object_transform", 9, 1.0),
    PrivilegedObsTerm("target_lift_pos", 3, 1.0),
    PrivilegedObsTerm("homie_commands", 7, 1.0),
)

PRIVILEGED_TEACHER_OBS_DIM = sum(term.dim for term in PRIVILEGED_TEACHER_OBS_SPEC)
PRIVILEGED_TEACHER_OBS_INFO_KEY = "_rlinf_privileged_teacher_obs"
PRIVILEGED_TEACHER_OBS_SLICES: dict[str, slice] = {}
_offset = 0
for _term in PRIVILEGED_TEACHER_OBS_SPEC:
    PRIVILEGED_TEACHER_OBS_SLICES[_term.name] = slice(_offset, _offset + _term.dim)
    _offset += _term.dim
del _offset
del _term

VIRAL_STAGE_NAMES = (
    "walk_to_object",
    "pre_place",
    "place",
    "grasp_and_lift",
    "turn",
)

TURNING_ON_RADIO_STAGE_TO_VIRAL_STAGE = {
    "move_to_radio": 0,  # walk_to_object
    "pickup_from_support": 3,  # grasp_and_lift
    "place_on_support": 2,  # place
    "press_radio": 4,  # turn
    "done": 4,
}


class BehaviorPrivilegedTeacherObsBuilder:
    """Builds the 226D VIRAL-style teacher observation for BEHAVIOR R1Pro.

    The exported layout is fixed by ``PRIVILEGED_TEACHER_OBS_SPEC``. This first
    implementation is intentionally conservative: signals that do not exist on
    the current R1Pro/OmniGibson interface are zero-filled with one-time warnings.
    """

    def __init__(self, logger=None):
        self.logger = logger
        self._warned: set[str] = set()

    def build(self, env, action=None) -> torch.Tensor:
        env = self._unwrap_env(env)
        robot = env.robots[0]
        action = self._as_tensor(action)
        components = {
            "base_lin_vel": self._base_lin_vel(robot),
            "base_ang_vel": self._base_ang_vel(robot),
            "projected_gravity": self._projected_gravity(robot),
            "actions": self._actions(action),
            "stage": self._stage(env),
            "delta_actions": self._delta_actions(action),
            "dof_pos": self._dof_pos(robot),
            "dof_vel": self._dof_vel(robot),
            "placement_pos": self._placement_pos(env),
            "table_pelvis_transform": self._table_pelvis_transform(env, robot),
            "hold_fingers_tips_force": self._fingertip_forces(
                "hold_fingers_tips_force"
            ),
            "hold_obj_transform": self._hold_obj_transform(env, robot),
            "hold_hand_object_transform": self._hold_hand_object_transform(env, robot),
            "target_place_pos": self._target_place_pos(env),
            "grasp_fingers_tips_force": self._fingertip_forces(
                "grasp_fingers_tips_force"
            ),
            "grasp_obj_transform": self._grasp_obj_transform(env, robot),
            "grasp_hand_object_transform": self._grasp_hand_object_transform(
                env, robot
            ),
            "target_lift_pos": self._target_lift_pos(env),
            "homie_commands": self._homie_commands(action),
        }

        pieces = []
        for term in PRIVILEGED_TEACHER_OBS_SPEC:
            value = components[term.name].to(dtype=torch.float32).flatten()
            assert value.numel() == term.dim, (
                f"Privileged obs term {term.name} expected dim {term.dim}, "
                f"got {value.numel()}"
            )
            pieces.append(value * term.scale)

        obs = torch.cat(pieces, dim=0)
        assert obs.numel() == PRIVILEGED_TEACHER_OBS_DIM, (
            f"Privileged teacher obs expected {PRIVILEGED_TEACHER_OBS_DIM}, "
            f"got {obs.numel()}"
        )
        assert torch.isfinite(obs).all(), "Privileged teacher obs contains NaN/Inf"
        return obs

    def summarize(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Returns scalar TensorBoard metrics for a single 226D privileged obs."""
        obs = self._cpu_tensor(obs).flatten()
        assert obs.numel() == PRIVILEGED_TEACHER_OBS_DIM, (
            f"Privileged teacher obs expected {PRIVILEGED_TEACHER_OBS_DIM}, "
            f"got {obs.numel()}"
        )
        metrics: dict[str, torch.Tensor] = {
            "privileged/enabled": torch.tensor(1.0),
            "privileged/obs_dim": torch.tensor(float(PRIVILEGED_TEACHER_OBS_DIM)),
            "privileged/finite": torch.isfinite(obs).all().to(dtype=torch.float32),
            "privileged/obs_l2": torch.linalg.vector_norm(obs),
            "privileged/obs_abs_mean": obs.abs().mean(),
            "privileged/obs_max_abs": obs.abs().max(),
        }

        terms = {
            name: obs[term_slice]
            for name, term_slice in PRIVILEGED_TEACHER_OBS_SLICES.items()
        }
        for term_name, value in terms.items():
            metrics[f"privileged/term_abs_mean/{term_name}"] = value.abs().mean()

        stage = terms["stage"]
        for idx, stage_name in enumerate(VIRAL_STAGE_NAMES):
            metrics[f"privileged/stage/{stage_name}"] = stage[idx]

        metrics.update(
            {
                "privileged/base_lin_vel_norm": torch.linalg.vector_norm(
                    terms["base_lin_vel"]
                ),
                "privileged/base_ang_vel_norm": torch.linalg.vector_norm(
                    terms["base_ang_vel"]
                ),
                "privileged/projected_gravity_norm": torch.linalg.vector_norm(
                    terms["projected_gravity"]
                ),
                "privileged/actions_abs_mean": terms["actions"].abs().mean(),
                "privileged/actions_max_abs": terms["actions"].abs().max(),
                "privileged/delta_actions_abs_mean": terms["delta_actions"]
                .abs()
                .mean(),
                "privileged/dof_pos_abs_mean": terms["dof_pos"].abs().mean(),
                "privileged/dof_vel_abs_mean": terms["dof_vel"].abs().mean(),
                "privileged/placement_xy_norm": torch.linalg.vector_norm(
                    terms["placement_pos"]
                ),
                "privileged/table_pelvis_distance": torch.linalg.vector_norm(
                    terms["table_pelvis_transform"][:3]
                ),
                "privileged/hold_obj_pelvis_distance": torch.linalg.vector_norm(
                    terms["hold_obj_transform"][:3]
                ),
                "privileged/hold_hand_object_distance": torch.linalg.vector_norm(
                    terms["hold_hand_object_transform"][:3]
                ),
                "privileged/target_place_z": terms["target_place_pos"][2],
                "privileged/grasp_obj_pelvis_distance": torch.linalg.vector_norm(
                    terms["grasp_obj_transform"][:3]
                ),
                "privileged/grasp_hand_object_distance": torch.linalg.vector_norm(
                    terms["grasp_hand_object_transform"][:3]
                ),
                "privileged/target_lift_z": terms["target_lift_pos"][2],
                "privileged/homie_command_norm": torch.linalg.vector_norm(
                    terms["homie_commands"]
                ),
            }
        )
        return metrics

    def _unwrap_env(self, env):
        while hasattr(env, "env"):
            env = env.env
        return env

    def _warn_once(self, key: str, message: str) -> None:
        if key in self._warned:
            return
        self._warned.add(key)
        if self.logger is not None:
            self.logger.warning(message)
        else:
            logging.getLogger(__name__).warning(message)

    @staticmethod
    def _as_tensor(value) -> torch.Tensor | None:
        if value is None:
            return None
        return BehaviorPrivilegedTeacherObsBuilder._cpu_tensor(value).flatten()

    @staticmethod
    def _cpu_tensor(value) -> torch.Tensor:
        if torch.is_tensor(value):
            return value.detach().cpu().to(dtype=torch.float32)
        return torch.as_tensor(value, dtype=torch.float32)

    @staticmethod
    def _pad_or_trim(value, dim: int) -> torch.Tensor:
        value = BehaviorPrivilegedTeacherObsBuilder._cpu_tensor(value).flatten()
        if value.numel() >= dim:
            return value[:dim]
        return torch.nn.functional.pad(value, (0, dim - value.numel()))

    @staticmethod
    def _quat_to_mat(quat: torch.Tensor) -> torch.Tensor:
        from omnigibson.utils import transform_utils as T

        return T.quat2mat(BehaviorPrivilegedTeacherObsBuilder._cpu_tensor(quat))

    @staticmethod
    def _relative_pose(pos, quat, frame_pos, frame_quat):
        from omnigibson.utils import transform_utils as T

        return T.relative_pose_transform(
            BehaviorPrivilegedTeacherObsBuilder._cpu_tensor(pos),
            BehaviorPrivilegedTeacherObsBuilder._cpu_tensor(quat),
            BehaviorPrivilegedTeacherObsBuilder._cpu_tensor(frame_pos),
            BehaviorPrivilegedTeacherObsBuilder._cpu_tensor(frame_quat),
        )

    @staticmethod
    def _rotation_6d(quat: torch.Tensor) -> torch.Tensor:
        rot = BehaviorPrivilegedTeacherObsBuilder._quat_to_mat(quat)
        return rot[:, :2].transpose(0, 1).reshape(-1)

    def _transform9_relative_to(self, obj, frame_pos, frame_quat) -> torch.Tensor:
        if obj is None:
            return torch.zeros(9, dtype=torch.float32)
        pos, quat = obj.get_position_orientation()
        rel_pos, rel_quat = self._relative_pose(pos, quat, frame_pos, frame_quat)
        return torch.cat([rel_pos, self._rotation_6d(rel_quat)], dim=0)

    @staticmethod
    def _proprio(robot) -> dict:
        return robot._get_proprioception_dict()

    def _base_lin_vel(self, robot) -> torch.Tensor:
        proprio = self._proprio(robot)
        _, base_quat = robot.get_position_orientation()
        return self._quat_to_mat(base_quat).T @ self._cpu_tensor(
            proprio["robot_lin_vel"]
        )

    def _base_ang_vel(self, robot) -> torch.Tensor:
        proprio = self._proprio(robot)
        _, base_quat = robot.get_position_orientation()
        return self._quat_to_mat(base_quat).T @ self._cpu_tensor(
            proprio["robot_ang_vel"]
        )

    def _projected_gravity(self, robot) -> torch.Tensor:
        _, base_quat = robot.get_position_orientation()
        gravity_world = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)
        return self._quat_to_mat(base_quat).T @ gravity_world

    def _actions(self, action: torch.Tensor | None) -> torch.Tensor:
        if action is None:
            return torch.zeros(31, dtype=torch.float32)
        if action.numel() != 31:
            self._warn_once(
                "actions_padded",
                "TODO(agent): R1Pro env action is not 31D; padding/trimming "
                "executed action to VIRAL actions dim 31.",
            )
        return self._pad_or_trim(action, 31)

    def _delta_actions(self, action: torch.Tensor | None) -> torch.Tensor:
        if action is None:
            return torch.zeros(11, dtype=torch.float32)
        action = self._pad_or_trim(action, 23)
        # R1Pro split: base[0:3], torso[3:7], left_arm[7:14],
        # left_gripper[14], right_arm[15:22], right_gripper[22].
        return torch.cat([action[0:3], action[15:22], action[22:23]], dim=0)

    def _dof_pos(self, robot) -> torch.Tensor:
        qpos = self._cpu_tensor(self._proprio(robot)["joint_qpos"])
        if qpos.numel() != 43:
            self._warn_once(
                "dof_pos_padded",
                "TODO(agent): R1Pro exposes fewer than VIRAL's 43 DoF positions; "
                "padding dof_pos to 43.",
            )
        return self._pad_or_trim(qpos, 43)

    def _dof_vel(self, robot) -> torch.Tensor:
        qvel = self._cpu_tensor(self._proprio(robot)["joint_qvel"])
        if qvel.numel() != 43:
            self._warn_once(
                "dof_vel_padded",
                "TODO(agent): R1Pro exposes fewer than VIRAL's 43 DoF velocities; "
                "padding dof_vel to 43.",
            )
        return self._pad_or_trim(qvel, 43)

    def _task_reward(self, env):
        reward_fns = getattr(getattr(env, "task", None), "_reward_functions", {})
        return reward_fns.get("task_specific")

    def _stage(self, env) -> torch.Tensor:
        reward = self._task_reward(env)
        stage_name = "move_to_radio"
        if reward is not None:
            stage_index = getattr(reward, "_stage_index", 0)
            stage_defs = getattr(reward, "_stage_defs", [])
            if stage_index < len(stage_defs):
                stage_name = stage_defs[stage_index].get("name", stage_name)
            else:
                stage_name = "done"
        stage_idx = TURNING_ON_RADIO_STAGE_TO_VIRAL_STAGE.get(stage_name, 0)
        stage = torch.zeros(5, dtype=torch.float32)
        stage[stage_idx] = 1.0
        return stage

    def _radio_and_table(self, env):
        reward = self._task_reward(env)
        radio = getattr(reward, "_radio_obj", None) if reward is not None else None
        table = getattr(reward, "_support_obj", None) if reward is not None else None
        if radio is None:
            radio = self._find_object_by_name(env, "radio_89")
        if table is None:
            table = self._find_object_by_name(env, "coffee_table_koagbh_0")
        if radio is None:
            self._warn_once(
                "radio_missing",
                "TODO(agent): Could not find turning_on_radio target radio; "
                "zero-filling radio-dependent privileged terms.",
            )
        if table is None:
            self._warn_once(
                "table_missing",
                "TODO(agent): Could not find turning_on_radio support table; "
                "zero-filling table placement privileged terms.",
            )
        return radio, table

    @staticmethod
    def _find_object_by_name(env, object_name: str):
        for obj in getattr(getattr(env, "scene", None), "objects", []):
            if obj is None or getattr(obj, "synset", None) == "agent":
                continue
            if object_name in {
                getattr(obj, "name", ""),
                getattr(obj, "prim_path", "").rsplit("/", 1)[-1],
            }:
                return obj
        return None

    def _aabb(self, obj):
        if obj is None:
            return None
        aabb = getattr(obj, "aabb", None)
        if aabb is not None:
            return aabb
        try:
            from omnigibson.object_states.aabb import AABB

            return obj.states[AABB].get_value()
        except Exception as exc:
            self._warn_once(
                f"aabb_missing_{getattr(obj, 'name', 'obj')}",
                f"TODO(agent): Could not query AABB for {getattr(obj, 'name', obj)} "
                f"({type(exc).__name__}); using object pose fallback.",
            )
            return None

    def _target_place_pos(self, env) -> torch.Tensor:
        radio, table = self._radio_and_table(env)
        if table is None:
            return torch.zeros(3, dtype=torch.float32)
        table_aabb = self._aabb(table)
        if table_aabb is None:
            return self._cpu_tensor(table.get_position_orientation()[0])
        low, high = table_aabb
        low = self._cpu_tensor(low)
        high = self._cpu_tensor(high)
        target = torch.stack(
            [(low[0] + high[0]) * 0.5, (low[1] + high[1]) * 0.5, high[2]]
        )
        radio_aabb = self._aabb(radio)
        if radio_aabb is not None:
            radio_low, radio_high = radio_aabb
            radio_low = self._cpu_tensor(radio_low)
            radio_high = self._cpu_tensor(radio_high)
            target[2] += (radio_high[2] - radio_low[2]) * 0.5
        return target

    def _placement_pos(self, env) -> torch.Tensor:
        return self._target_place_pos(env)[:2]

    def _table_pelvis_transform(self, env, robot) -> torch.Tensor:
        _, table = self._radio_and_table(env)
        root_pos, root_quat = robot.get_position_orientation()
        return self._transform9_relative_to(table, root_pos, root_quat)

    def _object_in_hand(self, robot):
        obj_in_hand = getattr(robot, "_ag_obj_in_hand", {})
        for arm in ("right", "left"):
            obj = obj_in_hand.get(arm)
            if obj is not None:
                return obj, arm
        return None, "right"

    def _hold_obj_transform(self, env, robot) -> torch.Tensor:
        obj, _ = self._object_in_hand(robot)
        root_pos, root_quat = robot.get_position_orientation()
        return self._transform9_relative_to(obj, root_pos, root_quat)

    def _hold_hand_object_transform(self, env, robot) -> torch.Tensor:
        obj, arm = self._object_in_hand(robot)
        hand_pos, hand_quat = robot.get_eef_pose(arm)
        return self._transform9_relative_to(obj, hand_pos, hand_quat)

    def _grasp_obj_transform(self, env, robot) -> torch.Tensor:
        radio, _ = self._radio_and_table(env)
        root_pos, root_quat = robot.get_position_orientation()
        return self._transform9_relative_to(radio, root_pos, root_quat)

    def _grasp_hand_object_transform(self, env, robot) -> torch.Tensor:
        radio, _ = self._radio_and_table(env)
        hand_pos, hand_quat = robot.get_eef_pose("right")
        return self._transform9_relative_to(radio, hand_pos, hand_quat)

    def _target_lift_pos(self, env) -> torch.Tensor:
        radio, _ = self._radio_and_table(env)
        if radio is None:
            return torch.zeros(3, dtype=torch.float32)
        pos = self._cpu_tensor(radio.get_position_orientation()[0]).clone()
        pos[2] += 0.25
        return pos

    def _fingertip_forces(self, key: str) -> torch.Tensor:
        # TODO(agent): R1Pro currently exposes gripper contact pairs/points, but
        # not VIRAL's thumb/index/middle/palm 3D force sensors. Keep the fixed
        # exported order and zero-fill until true per-tip force sensors are wired.
        self._warn_once(
            key,
            f"TODO(agent): {key} is unavailable on current R1Pro interface; "
            "zero-filling 4x3 fingertip force vector.",
        )
        return torch.zeros(12, dtype=torch.float32)

    def _homie_commands(self, action: torch.Tensor | None) -> torch.Tensor:
        commands = torch.zeros(7, dtype=torch.float32)
        if action is not None:
            # Current nearest HOMIE-like lower-body command is the 3D base
            # velocity action; the remaining four slots are reserved zeros.
            commands[:3] = self._pad_or_trim(action, 23)[:3]
        return commands
