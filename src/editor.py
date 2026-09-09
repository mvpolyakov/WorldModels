import argparse
import json
import copy
import genesis as gs
import genesis.vis.keybindings as kb
import torch
from scipy.spatial.transform import Rotation as R

from sim.scene_config import load_scene_config
from sim.build_world import WorldBuilder

class SceneEditor:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = load_scene_config(config_path)
        
        # In editor mode, force everything to be fixed so objects don't fall due to gravity
        for obj in self.config.landscape + self.config.mobiles:
            if 'fixed' not in obj.morph_params:
                obj.morph_params['fixed'] = True

        self.world, _ = WorldBuilder.scaffold_world(
            self.config,
            n_environments=1,
        )
        
        self.entities = []
        self.original_poses = {} # {name: (pos, quat)}
        
        # Gather all entities without relying on World.landscape
        # Creation order in scaffold_world: 0..N=Landscape
        landscape_configs = self.config.landscape
        for i, cfg in enumerate(landscape_configs):
            ent = self.world.scene.entities[i]
            self.entities.append((cfg.name, ent, cfg))
            
        for name, ent in self.world.mobiles.items():
            self.entities.append((name, ent, self._find_config(name)))
            
        self.selected_idx = 0
        self.step_size = 0.05
        self.rot_step_deg = 5.0
        
        # Record initial states
        for name, ent, cfg in self.entities:
            # get_pos / get_quat return tensors (1, 3) and (1, 4)
            pos = ent.get_pos()[0].cpu().numpy()
            quat = ent.get_quat()[0].cpu().numpy()
            self.original_poses[name] = (pos, quat)
            
        self._register_keybinds()
        self._highlight_selected()
        
    def _find_config(self, name):
        for obj in self.config.landscape + self.config.mobiles:
            if obj.name == name:
                return obj
        return None

    def _highlight_selected(self):
        print("\n" + "="*40)
        for i, (name, ent, cfg) in enumerate(self.entities):
            if i == self.selected_idx:
                print(f"[*] SELECTED: {name}")
                # Color change is tough dynamically in Genesis without recompiling shaders,
                # but we print a clear terminal UI for now.
            else:
                print(f"[ ] {name}")
        print("="*40)
        print("Controls: I/K (X axis), J/L (Y axis), U/O (Z axis). Hold SHIFT to rotate.")
        print("R to reset selected object to start. ENTER to save to JSON.")

    def cycle_selection(self):
        self.selected_idx = (self.selected_idx + 1) % len(self.entities)
        self._highlight_selected()

    def _get_current_pos_quat(self):
        name, ent, cfg = self.entities[self.selected_idx]
        pos = ent.get_pos()[0].cpu().numpy()
        quat = ent.get_quat()[0].cpu().numpy()
        return name, ent, cfg, pos, quat

    def move(self, dx, dy, dz):
        name, ent, cfg, pos, quat = self._get_current_pos_quat()
        pos[0] += dx
        pos[1] += dy
        pos[2] += dz
        ent.set_pos(pos)
        
    def rotate(self, droll, dpitch, dyaw):
        name, ent, cfg, pos, quat = self._get_current_pos_quat()
        # quat is [w, x, y, z] in Genesis
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]]) # scipy uses x,y,z,w
        delta = R.from_euler('xyz', [droll, dpitch, dyaw], degrees=True)
        new_r = r * delta
        new_q = new_r.as_quat()
        # back to w, x, y, z
        ent.set_quat([new_q[3], new_q[0], new_q[1], new_q[2]])

    def reset_selected(self):
        name, ent, cfg, pos, quat = self._get_current_pos_quat()
        orig_pos, orig_quat = self.original_poses[name]
        ent.set_pos(orig_pos)
        ent.set_quat(orig_quat)
        print(f"Reset '{name}' to original position.")

    def save_config(self):
        print("Saving current positions to config...")
        for name, ent, cfg in self.entities:
            pos = ent.get_pos()[0].cpu().numpy()
            quat = ent.get_quat()[0].cpu().numpy()
            
            # Genesis euler is xyz
            r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
            euler = r.as_euler('xyz', degrees=True)
            
            cfg.morph_params['pos'] = (float(pos[0]), float(pos[1]), float(pos[2]))
            # If the original used euler, update euler
            if 'euler' in cfg.morph_params or 'quat' not in cfg.morph_params:
                cfg.morph_params['euler'] = (float(euler[0]), float(euler[1]), float(euler[2]))
            else:
                cfg.morph_params['quat'] = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))

        for obj in self.config.mobiles:
            obj.morph_params.pop('fixed', None)
            
        # Save camera position
        cam_pos = self.world.scene.viewer.camera_pos
        cam_look = self.world.scene.viewer.camera_lookat
        self.config.camera.viewer_pos = (float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2]))
        self.config.camera.viewer_lookat = (float(cam_look[0]), float(cam_look[1]), float(cam_look[2]))
            
        self.config.to_json(self.config_path)
        print(f"Saved successfully to {self.config_path}!")

    def _register_keybinds(self):
        v = self.world.scene.viewer
        
        # Clear conflicting Genesis defaults for keys we want to use
        for key_to_remove in [kb.Key.I, kb.Key.J, kb.Key.K, kb.Key.L, kb.Key.U, kb.Key.O]:
            try:
                v.remove_keybind(key_to_remove)
            except Exception:
                pass
        
        # TAB to cycle
        v.register_keybinds(kb.Keybind("cycle", kb.Key.TAB, key_action=kb.KeyAction.PRESS, callback=self.cycle_selection))
        
        # R to reset
        v.register_keybinds(kb.Keybind("reset_sel", kb.Key.R, key_action=kb.KeyAction.PRESS, callback=self.reset_selected))
        
        # ENTER to save
        v.register_keybinds(kb.Keybind("save", kb.Key.ENTER, key_action=kb.KeyAction.PRESS, callback=self.save_config))

        # Translation (IJKL UO)
        binds = [
            ("mov_fwd", kb.Key.I, lambda: self.move(self.step_size, 0, 0)),
            ("mov_bwd", kb.Key.K, lambda: self.move(-self.step_size, 0, 0)),
            ("mov_lft", kb.Key.J, lambda: self.move(0, self.step_size, 0)),
            ("mov_rgt", kb.Key.L, lambda: self.move(0, -self.step_size, 0)),
            ("mov_up",  kb.Key.U, lambda: self.move(0, 0, self.step_size)),
            ("mov_dn",  kb.Key.O, lambda: self.move(0, 0, -self.step_size)),
        ]
        for name, key, cb in binds:
            v.register_keybinds(kb.Keybind(name, key, key_action=kb.KeyAction.PRESS, callback=cb))
            
        # Rotation (Number keys 1-6 instead of SHIFT, to avoid modifier bugs)
        rot_binds = [
            ("rot_x_pos", kb.Key.NUM_1, lambda: self.rotate(self.rot_step_deg, 0, 0)),
            ("rot_x_neg", kb.Key.NUM_3, lambda: self.rotate(-self.rot_step_deg, 0, 0)),
            ("rot_y_pos", kb.Key.NUM_4, lambda: self.rotate(0, self.rot_step_deg, 0)),
            ("rot_y_neg", kb.Key.NUM_6, lambda: self.rotate(0, -self.rot_step_deg, 0)),
            ("rot_z_pos", kb.Key.NUM_7, lambda: self.rotate(0, 0, self.rot_step_deg)),
            ("rot_z_neg", kb.Key.NUM_9, lambda: self.rotate(0, 0, -self.rot_step_deg)),
        ]
        for name, key, cb in rot_binds:
            v.register_keybinds(kb.Keybind(name, key, key_action=kb.KeyAction.PRESS, callback=cb))

    def run(self):
        print("Starting Editor loop... (Window should be open)")
        while True:
            # We step the world so viewer updates and position changes are rendered.
            # Because everything is 'fixed=True', physics won't cause them to fall.
            self.world.step()

def main():
    import os
    parser = argparse.ArgumentParser()
    default_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim", "configs", "simple_rolling.json")
    parser.add_argument("--config", type=str, default=default_config)
    args = parser.parse_args()

    gs.init(backend=gs.gpu, logging_level='warning')
    editor = SceneEditor(args.config)
    editor.run()

if __name__ == "__main__":
    main()
