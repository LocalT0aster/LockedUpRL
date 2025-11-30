# LockedUpRL

The game rules and agent description is located in [Rules.md](Rules.md)

## CLI

- `python main.py train --episodes 9000` — train catchers with DQN.
- `python main.py eval -n 300` — evaluate the saved model.
- `python main.py play [--runner astar|human]` — launch the pygame visualiser (runner prompt by default).
- `python main.py pipe --role catcher --catcher-index 0` — pipe mode for the Godot client.

### Pipe protocol

- Input: `stdin` supplies the current vision map; rows are space-separated codes from `Rules.md` (`_C`, `eC`, `oC`, `aeC`, `acC`, `unkC`). A blank line separates turns.
- Metadata (optional first line): space-delimited `role <token> pos <x> <y>`, e.g. `role C2 pos 3 12` or `role R pos 10 5`. Legacy `key=value` metadata is still accepted.
- Output: one line per turn in the form `<ROLE> act <verb> <dir?>`, e.g. `C2 act move up`, `R act stay`, `C1 act build left`.
