# Escape & Catch Game Environment

## Overview

This project defines a **two-agent game environment** where two types of agents interact on a field of discrete cells:

- **Escape Agent (Ae)** – The goal is to **escape through one of the exit cells**.
- **Catch Agent (Ac)** – The goal is to **catch or block Ae** before it escapes.

The environment is grid-based, turn-based, and supports multiple configurations and partial observability.

---

## Environment Description

### Field Structure

The environment consists of a **2D field of cells** with dimensions **H × W**.

Each cell has one of the following **states**:

| Code | Description | Movement Allowed |
|------|--------------|------------------|
| `_C` | Empty Cell | ✅ Yes |
| `eC` | Exit Cell | ✅ Yes — If Ae enters, Ae **wins** |
| `oC` | Obstacle Cell | ❌ No |
| `aeC` | Escape Agent Cell | ⛹️ Contains the Escape Agent |
| `acC` | Catch Agent Cell | 🧍 Contains a Catch Agent |
| `unkC` | Unknown Cell | 👁 Used for agents' partial knowledge |

---

### Environment Parameters

| Parameter | Description | Type | Default / Notes |
|------------|--------------|------|------------------|
| **H** | Field height | `int` | Required |
| **W** | Field width | `int` | Required |
| **eN** | Number of exit cells | `int` | Required |
| **oN** | Number of obstacle cells | `int` | Required |
| **aeN** | Number of escape agents | `int` | Constant = 1 |
| **acN** | Number of catch agents | `int` | Required |
| **tlN** | Maximum number of turns | `int` | Each turn = 1 Ae move + 1 Ac move |
| **con** | Connectivity | `int` | Constant = 4 (up, down, left, right) |

---

## Agents

### 1. Escape Agent (Ae)

#### Goal
The Escape Agent wins the game if it successfully reaches an **exit cell (`eC`)** before the turn limit or being caught.

#### Actions
Ae can perform **one** action per turn:

| Action | Description |
|---------|-------------|
| `move_up` | Move one cell up |
| `move_down` | Move one cell down |
| `move_left` | Move one cell left |
| `move_right` | Move one cell right |
| `stay` | Skip the turn (no movement) |

#### Parameters

| Parameter | Description | Type / Notes |
|------------|-------------|--------------|
| **eInfo** | Aware of number of exits | `bool` |
| **acInfo** | Aware of number of catch agents | `bool` |
| **oInfo** | Aware of initial number of obstacles | `bool` (default: `false`) |
| **FS** | Field size (H, W) | tuple `(int, int)` |
| **map** | Agent's perceived map (unknown cells are `unkC`) | 2D grid |
| **vision** | Vision distance (Chebyshev metric) | `int`, default = 1 |

**Vision Rule:**
After every action, all cells within the agent's vision range are updated on its map.

---

### 2. Catch Agent (Ac)

#### Goal
The Catch Agent's objective is to **stop Ae from escaping** — either by reaching the same cell as Ae or by building obstacles to trap Ae.

#### Actions
Each Catch Agent can perform **one** action per turn:

| Action | Description |
|---------|-------------|
| `move_up` | Move one cell up |
| `move_down` | Move one cell down |
| `move_left` | Move one cell left |
| `move_right` | Move one cell right |
| `stay` | Skip the turn (no movement) |
| `build_up` | Build an obstacle above if target cell is `_C` |
| `build_down` | Build an obstacle below if target cell is `_C` |
| `build_left` | Build an obstacle to the left if target cell is `_C` |
| `build_right` | Build an obstacle to the right if target cell is `_C` |

#### Parameters

| Parameter | Description | Type / Notes |
|------------|-------------|--------------|
| **eInfo** | Aware of number of exits | `bool` |
| **acInfo** | Aware of number of catch agents | `bool` |
| **oInfo** | Aware of initial number of obstacles | `bool` (default: `false`) |
| **FS** | Field size (H, W) | tuple `(int, int)` |
| **map** | Agent's perceived map (unknown cells are `unkC`) | 2D grid |
| **vision** | Vision distance (Chebyshev metric) | `int`, default = 1 |
| **order** | Action order (lower moves first) | `int` |

**Vision Rule:**
The catchers have the **shared map** of the world. It means that after every catcher action, all updated cells are updated for every catcher agent's map.

---

## Turn Structure

Each **turn** consists of two phases:

1. **Escape Phase:** The Escape Agent (Ae) chooses and executes one action.
2. **Catch Phase:** Each Catch Agent (Ac) executes its action in the order determined by `order`.

After each turn:
- All agents' vision updates occur.
- The environment checks for **win conditions** or **turn limit**.

---

## Win Conditions

| Condition | Winner |
|------------|---------|
| Ae enters `eC` cell | **Escape Agent (Ae)** wins |
| Ac occupies the same cell as Ae | **Catch Agents (Ac)** win |
| Turn limit (`tlN`) is reached without a winner | **Draw** |

---

## Connectivity and Movement Rules

- Movement is limited to **4-connected neighbors** (up, down, left, right).
- Agents cannot move into cells with state `oC` (obstacle).
- Agents cannot move outside the field boundaries.
- Multiple Catch Agents may not occupy the same cell.

---

## Vision and Perception

- "Fog of war" kind of vision of the **whole** map.
- Both Ae and Ac agents **see** and **perceive** cells within their **vision radius** using the **Chebyshev distance metric**.
- All **perceived** cells are updated in the **agent's map** each turn.
- **Seen** cells **remain** visible, but are **not updated unless perceived**.
- **Unseen** cells remain **unknown (`unkC`)**.

---

## Example

```
Field (5x5)
---------------------
_C  _C  eC  _C  _C
_C  oC  _C  _C  _C
_C  _C  aeC _C  _C
_C  _C  _C  acC _C
_C  _C  _C  _C  _C
---------------------
```

- Ae must reach `eC` to win.
- Ac can move or build obstacles to block Ae's path.

---

## Customization

You can modify the following to create different scenarios:
- Field size (H, W)
- Number and position of exits (`eN`)
- Number and position of obstacles (`oN`)
- Number of catch agents (`acN`)
- Maximum turns (`tlN`)
- Vision distance for each agent

---

## Pipe Metadata and Action Strings

When using the pipe interface (e.g., with Godot), each turn can include a metadata line before the grid rows:

- **Meta line (space-delimited)**: `role <R|R1|C1|C2|...> pos <x> <y>`
  - `role` — runner (`R`/`R1`) or catcher index (`C1`, `C2`, ...; 1-based for readability).
  - `pos` — the agent’s current coordinates as `x y`.
  - Legacy `key=value` metadata is still accepted.

- **Action line (emitted per turn)**: `<ROLE> act <verb> <dir?>`
  - Movement: `move up|down|left|right|stay`
  - Build (catchers): `build up|down|left|right`
  - Examples: `C2 act move up`, `R act stay`, `C1 act build left`
