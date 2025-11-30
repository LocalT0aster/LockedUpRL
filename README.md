# LockedUpRL

The game rules and agent description is located in [Rules.md](Rules.md)

## CLI

- `python main.py train --episodes 9000` — train catchers with DQN.
- `python main.py eval -n 300` — evaluate the saved model.
- `python main.py play [--runner astar|human]` — launch the pygame visualiser (runner prompt by default).
- `python main.py pipe --role catcher --catcher-index 0` — pipe mode for the Godot client.

### Pipe protocol

`stdin` supplies the current vision map; rows are space-separated codes from `Rules.md` (`_C`, `eC`, `oC`, `aeC`, `acC`, `unkC`). A blank line separates turns. An optional first line can hold metadata, e.g. `role=runner id=1`. The script prints one action per turn to `stdout`:

- Runner actions: `move_up`, `move_down`, `move_left`, `move_right`, `stay`
- Catcher actions: `move_*`, `build_*`, `stay`
