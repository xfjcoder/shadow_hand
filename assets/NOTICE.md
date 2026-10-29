`right_hand.xml`, `scene_right.xml`, `keyframes.xml`, and everything under `assets/` are vendored
unmodified from [`google-deepmind/mujoco_menagerie`](https://github.com/google-deepmind/mujoco_menagerie)'s
`shadow_hand` package (Shadow Hand E3M5; local checkout `~/robo/mujoco_menagerie`, last commit
touching the package `71f066a`). The original URDF and meshes were provided by Shadow Robot Company
under the Apache-2.0 License (`LICENSE.shadow_hand`; see `CHANGELOG.shadow_hand.md` and that
package's README for its URDF-to-MJCF derivation steps). Only the right hand was copied.

Everything else in this repo (scenes and tasks built on top of this model, controllers, training
code, docs) is original to this project.
